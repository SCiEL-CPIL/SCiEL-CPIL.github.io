"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed set of non-linear basis functions.
    The 10 params are output weights for these basis functions.
    
    Basis functions are designed to cover a wide range of functional forms:
    - Linear terms
    - Polynomial terms (up to degree 3)
    - Trigonometric terms
    - Exponential-like terms via tanh/sigmoid
    - Interaction terms
    """
    # Ensure x, y are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Clip params to avoid overflow in non-linear functions
    p = np.clip(params, -1e6, 1e6)
    
    # Fixed non-linear basis functions (10 total)
    # Each basis function is a scalar per point, so we compute feature matrix F (n, 10)
    # Then prediction = F @ p
    
    # Basis 1: constant (intercept)
    b1 = np.ones_like(x)
    
    # Basis 2: x
    b2 = x
    
    # Basis 3: y
    b3 = y
    
    # Basis 4: x*y (interaction)
    b4 = x * y
    
    # Basis 5: x^2
    b5 = x * x
    
    # Basis 6: y^2
    b6 = y * y
    
    # Basis 7: tanh(x)
    b7 = np.tanh(x)
    
    # Basis 8: tanh(y)
    b8 = np.tanh(y)
    
    # Basis 9: tanh(x + y)
    b9 = np.tanh(x + y)
    
    # Basis 10: tanh(x - y)
    b10 = np.tanh(x - y)
    
    # Stack into feature matrix (n, 10)
    F = np.column_stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10])
    
    # Prediction: linear combination of basis functions
    pred = F @ p
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
