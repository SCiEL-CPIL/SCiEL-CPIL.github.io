"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of 5 Gaussian radial basis functions with non-linear parameterization.
    
    Parameters (10 total):
      p[0:5]  - amplitudes of the 5 Gaussians
      p[5:10] - means (centers) of the 5 Gaussians
    
    Widths are fixed at 0.5 to keep the problem well-conditioned for BFGS
    while still allowing localized features.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    amplitudes = p[0:5]
    centers = p[5:10]
    
    # Fixed widths for stability
    widths = np.full(5, 0.5)
    
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(5):
        diff = x - centers[i]
        y += amplitudes[i] * np.exp(-(diff * diff) / (2.0 * widths[i] * widths[i]))
    
    return y
# EVOLVE-BLOCK-END
