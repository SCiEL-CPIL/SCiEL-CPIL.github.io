"""
Radial Basis Function (RBF) network with a 3x3 grid of fixed centers in [-1,1]^2.
Gaussian kernels provide localized, non-linear basis functions that can capture
complex smooth structures (periodic, multi-scale) that global polynomials miss.
The 10th parameter serves as a global offset. Width is tuned for the grid spacing.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed 3x3 grid of centers in [-1, 1]^2
    cx = np.array([-1.0, 0.0, 1.0])
    cy = np.array([-1.0, 0.0, 1.0])
    
    # Gaussian kernel width (controls locality)
    width = 0.8
    
    # Pre-compute RBF activations for each center
    # p0..p8: weights for the 9 RBF centers
    # p9: global offset
    
    result = params[9] * np.ones_like(x)
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            # Distance squared from (x, y) to center (cx[i], cy[j])
            dx = x - cx[i]
            dy = y - cy[j]
            r2 = dx * dx + dy * dy
            # Gaussian kernel
            kernel = np.exp(-r2 / (2.0 * width * width))
            result += params[idx] * kernel
    
    return result
# EVOLVE-BLOCK-END
