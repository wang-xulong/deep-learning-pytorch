# softmax与分类模型

## 知识点

### 为什么要用softmax?

softmax函数是来自于sigmoid函数在多分类情况下的推广，他们的相同之处：（凌云峰）

1.都具有良好的数据压缩能力是实数域R→[ 0 , 1 ]的映射函数，可以将杂乱无序没有实际含义的数字直接转化为每个分类的可能性概率。

2.都具有非常漂亮的导数形式，便于反向传播计算。

3.它们都是 soft version of max ，都可以将数据的差异明显化。

相同的，他们具有着不同的特点，sigmoid函数可以看成softmax函数的特例，softmax函数也可以看作sigmoid函数的推广。

1.sigmoid函数前提假设是样本服从伯努利 (Bernoulli) 分布的假设，而softmax则是基于多项式分布。首先证明多项分布属于指数分布族，这样就可以使用广义线性模型来拟合这个多项分布，由广义线性模型推导出的目标函数即为Softmax回归的分类模型。 

2.sigmoid函数用于分辨每一种情况的可能性，所以用sigmoid函数实现多分类问题的时候，概率并不是归一的，反映的是每个情况的发生概率，因此非互斥的问题使用sigmoid函数可以获得比较漂亮的结果；softmax函数最初的设计思路适用于首先数字识别这样的互斥的多分类问题，因此进行了归一化操作，使得最后预测的结果是唯一的。

### 为什么要使用交叉熵损失函数？

使用平方损失函数是可以的，只是其要求太过严格，从极大似然估计的角度来看，他要求每个样本符合独立同分布的假设。改善上述问题的一个方法是使用更适合两个概率分布差异的测量函数。

逻辑回归配合MSE损失函数时，采用梯度下降法进行学习时，会出现模型一开始训练时，学习速率非常慢的情况。

https://zhuanlan.zhihu.com/p/35709485





Pytorch.gather()的用法

给出定义：torch.gather(input, dim, index, out=None) → Tensor

官方解释：沿给定轴dim，将输入索引张量index指定位置的值进行聚合。

 torch.gather(input, dim, index, out=None)中的dim表示的就是第几维。For example, 如果dim=0，

那么它表示的就是你接下来的操作是对于第一维度进行的，也就是行；如果dim=1,那么它表示的就是你接下来的操作是对于第二维度进行的，也就是列。

代码示例

```python
b = torch.Tensor([[1,2,3],[4,5,6]])
print b
index_1 = torch.LongTensor([[0,1],[2,0]])
index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
print torch.gather(b, dim=1, index=index_1)
print torch.gather(b, dim=0, index=index_2)
```

输出结果

```python
 1  2  3
 4  5  6
[torch.FloatTensor of size 2x3]

 1  2
 6  4
[torch.FloatTensor of size 2x2]

 1  5  6
 1  2  3
[torch.FloatTensor of size 2x3]
```

