# Experiment-2--Implementation-of-Perceptron
##AIM:

To implement a perceptron for classification using Python

EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one 
f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’

A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.


ALGORITHM:
Importing the libraries
Importing the dataset
Plot the data to verify the linear separable dataset and consider only two classes
Convert the data set to scale the data to uniform range by using Feature scaling

Split the dataset for training and testing
Define the input vector ‘X’ from the training dataset
Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2
Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
Assign the learning rate
For ‘N ‘ iterations ,do the following:
        v(i) = w(i)*x(i)
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)
Plot the error for each iteration 
Print the accuracy


 PROGRAM:
 
 V.S.Sundhararajan 
 212220040165
 
 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
def __init__(self, learning_rate=0.1):
 self.learning_rate = learning_rate
 self._b = 0.0  # y-intercept
 self._w = None  # weights assigned to input features
 self.misclassified_samples = []
def fit(self, x: np.array, y: np.array, n_iter=10):
 self._b = 0.0
 self._w = np.zeros(x.shape[1])
 self.misclassified_samples = []
 for _ in range(n_iter):
   # counter of the errors during this training iteration
   errors = 0
   for xi, yi in zip(x, y):
     update = self.learning_rate * (yi - self.predict(xi))
     self._b += update
     self._w += update * xi
     errors += int(update != 0.0)
   self.misclassified_samples.append(errors)
def f(self, x: np.array) -> float:
 return np.dot(x, self._w) + self._b
def predict(self, x: np.array):
 return np.where(self.f(x) >= 0, 1, -1)
df = pd.read_csv('/content/IRIS.csv')
df.head()
# extract the label column
y = df.iloc[:, 4].values
# extract features
x = df.iloc[:, 0:3].values

x = x[0:100, 0:2]  # reduce the dimensionality of the data
y = y[0:100]
# plot Iris Setosa samples
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
# plot Iris Versicolour samples
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
         label='Versicolour')
# show the legend
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
# show the plot
plt.show()
# map the labels to a binary integer value
y = np.where(y == 'Iris-setosa', 1, -1)


x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()


# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)
# train the model
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)
print("accuracy" , accuracy_score(classifier.predict(x_test), y_test)*100)
# plot the number of errors during each iteration
plt.plot(range(1, len(classifier.misclassified_samples) + 1),classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()


Output:
![196998934-8c45bcd6-f800-44c5-aade-94442107935e](https://user-images.githubusercontent.com/63903041/236727117-12a5aa13-0dda-473d-aa30-441e88b2e687.jpg)
![196998986-96762d88-33f0-460a-a901-5df21ba58711](https://user-images.githubusercontent.com/63903041/236727139-0681ef5f-a337-4d53-8cfd-9a1d1c6219c4.jpg)
![196999009-5c2b1c5c-1c45-4f9c-a9d9-58e585f4b9d8](https://user-images.githubusercontent.com/63903041/236727167-6d5832ea-fd7d-4031-85c7-a2f382d52b45.jpg)

