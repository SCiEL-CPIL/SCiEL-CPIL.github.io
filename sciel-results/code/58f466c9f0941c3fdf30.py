"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a hybrid basis of global trends and local
    Radial Basis Functions (RBFs).
    
    Basis (10 terms):
    b0 = 1
    b1 = x
    b2 = sin(x)
    b3 = cos(x)
    b4 = exp(-x)
    b5 = exp(-(x - 1.5)^2)
    b6 = exp(-(x - 2.5)^2)
    b7 = exp(-(x - 3.5)^2)
    b8 = exp(-(x - 4.5)^2)
    b9 = exp(-(x - 5.5)^2)
    
    This basis is designed to be well-conditioned and flexible:
    - Global terms (1, x, sin, cos, exp) capture overall trends and periodicity.
    - Local RBF terms allow the model to fit specific features, peaks, or
      non-smooth behaviors that global functions cannot capture.
    - All basis functions are bounded or slowly varying, which helps BFGS
      converge from the all-ones start.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
        
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.05)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Global Basis
    b0 = 1.0
    b1 = x
    b2 = np.sin(x)
    b3 = np.cos(x)
    b4 = np.exp(-x)
    
    # Local Radial Basis Functions (Gaussian kernels)
    # Centers distributed across the input range [0.1, 6]
    # Width sigma = 1.0 (implicit in exponent - (x-c)^2)
    b5 = np.exp(-(x - 1.5) ** 2)
    b6 = np.exp(-(x - 2.5) ** 2)
    b7 = np.exp(-(x - 3.5) ** 2)
    b8 = np.exp(-(x - 4.5) ** 2)
    b9 = np.exp(-(x - 5.5) ** 2)
    
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
