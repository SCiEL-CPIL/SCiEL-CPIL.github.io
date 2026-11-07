# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Hybrid model combining a fixed Fourier basis and a small neural network.
    The 10 parameters are split between:
    - 6 parameters for a Fourier-like basis: constant, sin(x), cos(x), sin(y), cos(y), sin(x+y)
    - 4 parameters for a single-neuron tanh network: w_x, w_y, w_out, bias
    
    This allows the model to capture both periodic/oscillatory components (via Fourier terms)
    and general non-linear interactions (via the tanh neuron). The BFGS optimizer will fit
    all 10 parameters jointly to minimize MSE.
    """
    p = params
    
    # Fourier basis terms (6 params)
    # f0: constant
    # f1: sin(x)
    # f2: cos(x)
    # f3: sin(y)
    # f4: cos(y)
    # f5: sin(x+y)
    
    f0 = 1.0
    f1 = np.sin(x)
    f2 = np.cos(x)
    f3 = np.sin(y)
    f4 = np.cos(y)
    f5 = np.sin(x + y)
    
    # Neural network term (4 params)
    # h = tanh(w_x * x + w_y * y + bias)
    # out = w_out * h
    
    w_x = p[6]
    w_y = p[7]
    bias = p[8]
    w_out = p[9]
    
    h = np.tanh(w_x * x + w_y * y + bias)
    nn_out = w_out * h
    
    # Combine Fourier basis and neural network output
    pred = (p[0] * f0 + p[1] * f1 + p[2] * f2 +
            p[3] * f3 + p[4] * f4 + p[5] * f5 + nn_out)
    
    return pred
# EVOLVE-BLOCK-END
