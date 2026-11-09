"""Symbolic regression with 1 input using a rich basis of functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a linear combination of 10 basis functions.
    
    Basis functions (coefficients are params[0..9]):
    0: constant (1)
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: exp(-x^2)  (Gaussian bump, well-behaved on [-3,3])
    7: tanh(x)
    8: x * sin(x)
    9: x * cos(x)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    # Compute each basis function
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.exp(-x ** 2)
    b7 = np.tanh(x)
    b8 = x * np.sin(x)
    b9 = x * np.cos(x)
    
    # Linear combination
    result = (params[0] * b0 +
              params[1] * b1 +
              params[2] * b2 +
              params[3] * b3 +
              params[4] * b4 +
              params[5] * b5 +
              params[6] * b6 +
              params[7] * b7 +
              params[8] * b8 +
              params[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
