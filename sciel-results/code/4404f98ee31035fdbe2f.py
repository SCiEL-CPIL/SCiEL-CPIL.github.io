import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x and y are 1D arrays of shape (N,)
    # params is 1D array of shape (10,)
    
    # Ensure x and y are 1D for broadcasting
    x = x.flatten()
    y = y.flatten()
    
    # Mixed basis: Polynomial, Trigonometric, Radial
    # 1. Constant
    X1 = np.ones_like(x)
    # 2. Linear
    X2 = x
    X3 = y
    # 3. Quadratic
    X4 = x**2
    X5 = y**2
    X6 = x * y
    # 4. Trigonometric
    X7 = np.sin(np.pi * x)
    X8 = np.sin(np.pi * y)
    # 5. Radial / Localized
    X9 = np.exp(-(x**2 + y**2))
    # 6. Interaction Trig
    X10 = np.cos(np.pi * x) * np.cos(np.pi * y)
    
    # Stack into a matrix of shape (N, 10)
    features = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10))
    
    # Linear combination
    return features @ params
