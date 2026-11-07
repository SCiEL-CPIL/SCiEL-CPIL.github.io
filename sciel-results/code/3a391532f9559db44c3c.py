# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x to [-1, 1] for numerical stability of high-degree polynomials
    # x range is [0.1, 6.0]
    x_min = 0.1
    x_max = 6.0
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 2.0
    
    u = (np.asarray(x, dtype=np.float64) - center) / scale
    
    # Compute polynomial basis up to degree 9
    # We have 10 parameters, so we can fit a degree 9 polynomial
    # p0 + p1*u + ... + p9*u^9
    
    # Use np.vander for efficient computation of powers
    # vander(u, N=10, increasing=True) gives columns [1, u, u^2, ..., u^9]
    basis = np.vander(u, N=10, increasing=True)
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
