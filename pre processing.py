import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
from glob import glob
import os, shutil
from tqdm.notebook import tqdm
tqdm.pandas()

from IPython import display as ipd
from joblib import Parallel, delayed

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf

#这部分内容为Run LengthEncoding行程编码的解码
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

#为Run LengthEncoding行程编码的编码
# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



#从输入的数据行中提取和解析元数据信息（文件名处理），将文件名转化为信息
def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row
#从路径中提取和解析相关数据信息（文件路径处理）
def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row
#将已有的RLE信息和ID转化为对应图像的蒙版（可视化处理）
def id2mask(id_):
    idf = df[df['id']==id_]
    wh = idf[['height','width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
        cdf = idf[idf['class']==class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask
#RGB通道转灰度
def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0,0),(0,0),(1,0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask
#灰度转RGB通道
def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)
"""加载图片，并进行一些必要的处理以便后续使用。
首先，函数调用 OpenCV 库（cv2）中的 imread 函数来读取给定路径（path）的图像，同时设置 cv2.IMREAD_UNCHANGED 参数以保持图像的原始深度和通道数。
读取完图片后，通过 astype 函数将图像数据类型从 uint16（原始类型）转换为 float32。这是为了能进行下一步的归一化操作，因为归一化涉及到浮点数的运算。
归一化操作是将图片的像素值从它们的原始范围（可能是0到65535，因为原始数据类型是 uint16）缩放至0到255的范围。这是通过线性变换实现的，线性变换的公式是：(img - img.min()) / (img.max() - img.min()) * 255.0。
最后，函数再次调用 astype 函数，将归一化后的图像数据类型从 float32 转回为 uint8。这样做是为了保存图像时可以减少存储空间，同时也是因为大多数图像处理的库和函数都需要 uint8 类型的输入。
在完成所有这些处理后，函数返回处理过的图像。"""
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32') # original is uint16
    img = (img - img.min())/(img.max() - img.min())*255.0 # scale image to [0, 255]
    img = img.astype('uint8')
    return img
"""显示输入的图像，如果存在蒙版，则叠加显示蒙版。它采用了一种数据增强技巧叫做直方图均衡化，并利用一种称为CLAHE（对比度受限的自适应直方图均衡化）的方法来增强图像的对比度。
函数开头，创建了一个 clahe 对象，其中 clipLimit 参数控制了对比度的增强程度，而 tileGridSize 参数设定了均衡化计算的区域大小。然后使用 clahe 对象的 apply 方法将这种增强应用到输入的图像上。
然后，使用 Matplotlib 库的 imshow 函数显示增强后的图像，图像的颜色映射（colormap）使用了 'bone'。
如果提供了蒙版，函数使用相同的 imshow 函数，以半透明方式（alpha=0.5）打印蒙版于原图上。并放置了一个图例，表示 "Large Bowel", "Small Bowel", "Stomach"三种不同的医学结构的蒙版颜色。
最后，用 plt.axis('off')隐藏坐标轴。"""
def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = [ "Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')
#保存生成的蒙版
def save_mask(id_):
    idf = df[df['id']==id_]
    mask = id2mask(id_)*255
    image_path = idf.image_path.iloc[0]
    mask_path = image_path.replace('/kaggle/input/','/tmp/png/')
    mask_folder = mask_path.rsplit('/',1)[0]
    os.makedirs(mask_folder, exist_ok=True)
    cv2.imwrite(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    mask_path2 = image_path.replace('/kaggle/input/','/tmp/np/').replace('.png','.npy')
    mask_folder2 = mask_path2.rsplit('/',1)[0]
    os.makedirs(mask_folder2, exist_ok=True)
    np.save(mask_path2, mask)
    return mask_path    

#读取文件   
df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
# df = df[~df.segmentation.isna()]
df = df.progress_apply(get_metadata, axis=1)
df.head()

paths = glob('/kaggle/input/uw-madison-gi-tract-image-segmentation/train/*/*/*/*')
path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.progress_apply(path2info, axis=1)
df = df.merge(path_df, on=['case','day','slice'])
df.head()
#选取并处理相关图像
row=1; col=4
plt.figure(figsize=(5*col,5*row))
for i, id_ in enumerate(df[~df.segmentation.isna()].sample(frac=1.0)['id'].unique()[:row*col]):
    img = load_img(df[df['id']==id_].image_path.iloc[0])
    mask = id2mask(id_)*255
    plt.subplot(row, col, i+1)
    i+=1
    show_img(img, mask=mask)
    plt.tight_layout()
    
ids = df['id'].unique()
_ = Parallel(n_jobs=-1, backend='threading')(delayed(save_mask)(id_)\
                                             for id_ in tqdm(ids, total=len(ids)))

i = 250
img = load_img(df.image_path.iloc[i])
mask_path = df['image_path'].iloc[i].replace('/kaggle/input/','/tmp/png/')
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
plt.figure(figsize=(5,5))
show_img(img, mask=mask)
#保存数据
df['mask_path'] = df.image_path.str.replace('/kaggle/input','/kaggle/input/uwmgi-mask-dataset/png/')
df.to_csv('train.csv',index=False)
#压缩打包
shutil.make_archive('/kaggle/working/png',
                    'zip',
                    '/tmp/png',
                    'uw-madison-gi-tract-image-segmentation')
