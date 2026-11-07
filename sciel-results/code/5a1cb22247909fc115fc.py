"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 2 kernels in 3D.
    10 params: [w1, w2, c1x, c1y, c1z, s1, c2x, c2y, c2z, s2]
    """
    p = params
    # Extract parameters
    w1 = p[0]
    w2 = p[1]
    c1x, c1y, c1z = p[2], p[3], p[4]
    s1 = p[5]
    c2x, c2y, c2z = p[6], p[7], p[8]
    s2 = p[9]
    
    # Ensure widths are positive and smooth for BFGS
    s1_safe = np.sqrt(s1**2 + 1e-6)
    s2_safe = np.sqrt(s2**2 + 1e-6)
    
    # Compute squared distances
    d1_sq = (x - c1x)**2 + (y - c1y)**2 + (z - c1z)**2
    d2_sq = (x - c2x)**2 + (y - c2y)**2 + (z - c2z)**2
    
    # Compute Gaussian kernels
    k1 = np.exp(-d1_sq / (2.0 * s1_safe**2))
    k2 = np.exp(-d2_sq / (2.0 * s2_safe**2))
    
    return w1 * k1 + w2 * k2
# EVOLVE-BLOCK-END
