"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based model with 5 kernels at symmetric points, 2 anisotropic widths,
    and 3 linear bias terms. Total 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has at least 10 elements; pad with zeros if needed
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    p = p[:10]
    
    # Extract parameters
    w1, w2, w3, w4, w5 = p[0], p[1], p[2], p[3], p[4]
    sigma_x = p[5]
    sigma_y = p[6]
    b0, b1, b2 = p[7], p[8], p[9]
    
    # Ensure positive widths for numerical stability
    sigma_x = np.maximum(sigma_x, 0.05)
    sigma_y = np.maximum(sigma_y, 0.05)
    
    # Kernel centers
    centers = np.array([[0.0, 0.0], 
                        [0.5, 0.0], 
                        [-0.5, 0.0], 
                        [0.0, 0.5], 
                        [0.0, -0.5]])
    
    # Compute anisotropic RBF values
    pred = np.zeros_like(x)
    weights = np.array([w1, w2, w3, w4, w5])
    
    for i in range(5):
        dx = x - centers[i, 0]
        dy = y - centers[i, 1]
        dist_sq = (dx / sigma_x) ** 2 + (dy / sigma_y) ** 2
        kernel_val = np.exp(-0.5 * dist_sq)
        pred += weights[i] * kernel_val
    
    # Add linear bias terms
    pred += b0 + b1 * x + b2 * y
    
    return pred
# EVOLVE-BLOCK-END
