"""
Neural Network Basis Strategy.

The previous strategy used a fixed linear combination of Fourier terms. 
This strategy uses a small Multi-Layer Perceptron (MLP) where the 10 parameters
are distributed between weights and biases of a 2-layer network.

Architecture:
- Input: 2 features (x, y)
- Hidden Layer 1: 2 neurons (Tanh activation)
  - Weights: 2x2 = 4 params (p[0:4])
  - Biases: 2 params (p[4:6])
- Hidden Layer 2: 1 neuron (Tanh activation)
  - Weights: 1x2 = 2 params (p[6:8])
  - Biases: 1 param (p[8])
- Output Layer: 1 neuron (Linear activation)
  - Weights: 1 param (p[9])
  - Bias: Fixed to 0 (or could use another param, but we are limited to 10)
  
Wait, 4+2+2+1+1 = 10 params.
Let's verify:
Input dim: 2
H1 dim: 2
W1: 2x2 = 4
b1: 2
H2 dim: 1
W2: 1x2 = 2
b2: 1
Output dim: 1
W3: 1
b3: 0 (fixed)
Total: 4+2+2+1+1 = 10.

This allows the model to learn non-linear interactions and shapes that a simple
linear Fourier basis might struggle with if the function is not purely periodic
or has complex interactions. The non-linearity (Tanh) provides local adaptability.
"""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are 1D arrays for consistent indexing
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Extract parameters
    # W1 (2x2), b1 (2), W2 (2x1), b2 (1), W3 (1)
    W1 = params[0:4].reshape(2, 2)
    b1 = params[4:6]
    W2 = params[6:8].reshape(2, 1)
    b2 = params[8:9]
    W3 = params[9:10]
    
    # Input layer
    # X is (N, 2)
    X = np.column_stack((x, y))
    
    # Hidden Layer 1
    Z1 = X @ W1.T + b1  # (N, 2)
    A1 = np.tanh(Z1)
    
    # Hidden Layer 2
    Z2 = A1 @ W2.T + b2  # (N, 1)
    A2 = np.tanh(Z2)
    
    # Output Layer (Linear)
    # A2 is (N, 1), W3 is (1,). 
    # We want (N, 1) @ (1, 1).T -> (N, 1)
    # Or just dot product.
    pred = A2 @ W3.T
    
    # Flatten to 1D
    return pred.flatten()
