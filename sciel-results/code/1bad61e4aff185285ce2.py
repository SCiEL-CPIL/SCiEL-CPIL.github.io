"""Symbolic regression with 1 input(s) using a rich mixed basis with RBF components."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using a mixed basis: polynomial, trigonometric, exponential,
    logarithmic, tanh, and Gaussian RBF components. Linear in parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = np.asarray(params, dtype=np.float64)
    
    # Clip x for safety in log and other functions
    x_safe = np.clip(x, 1e-10, None)
    
    # Basis functions:
    # b0 = 1 (constant)
    # b1 = x
    # b2 = x^2
    # b3 = x^3
    # b4 = exp(x)
    # b5 = exp(-x)
    # b6 = sin(x)
    # b7 = cos(x)
    # b8 = log(x)
    # b9 = tanh(x)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = np.exp(x)
    b5 = np.exp(-x)
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.log(x_safe)
    b9 = np.tanh(x)
    
    # Linear combination
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
