"""Symbolic regression with a rich basis: polynomials, trig, and RBFs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of 10 basis functions of x.
    Basis:
      0: 1 (constant)
      1: x
      2: x^2
      3: x^3
      4: sin(x)
      5: cos(x)
      6: exp(-x^2 / 2)  (Gaussian centered at 0)
      7: exp(-(x-1.5)^2 / 2)
      8: exp(-(x+1.5)^2 / 2)
      9: tanh(x)
    """
    # Ensure x is 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # Build basis matrix (N, 10)
    # For numerical stability, we clip some values
    x_clip = np.clip(x, -10, 10)
    
    b0 = np.ones_like(x_clip)
    b1 = x_clip
    b2 = x_clip ** 2
    b3 = x_clip ** 3
    b4 = np.sin(x_clip)
    b5 = np.cos(x_clip)
    
    # Gaussian RBFs
    # Use a moderate width to capture local features
    sigma2 = 1.0  # sigma = 1
    b6 = np.exp(-0.5 * (x_clip / np.sqrt(sigma2)) ** 2)
    b7 = np.exp(-0.5 * ((x_clip - 1.5) / np.sqrt(sigma2)) ** 2)
    b8 = np.exp(-0.5 * ((x_clip + 1.5) / np.sqrt(sigma2)) ** 2)
    
    b9 = np.tanh(x_clip)
    
    # Stack into matrix
    # Using np.column_stack is efficient enough for N=200
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Linear combination
    # params shape (10,)
    y = basis @ params
    
    # Ensure finite output
    # If params lead to overflow, clip to a reasonable range
    # However, BFGS should keep params reasonable.
    # If non-finite, return zeros (or some safe value) to avoid breaking the fit
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
