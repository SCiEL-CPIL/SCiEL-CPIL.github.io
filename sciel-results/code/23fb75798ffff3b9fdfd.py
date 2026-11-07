"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 2 kernels in 3D + bias.
    
    params layout (10 total):
      p[0]     = bias (constant offset)
      p[1:4]   = center of kernel 1 (cx1, cy1, cz1)
      p[4]     = weight of kernel 1
      p[5:8]   = center of kernel 2 (cx2, cy2, cz2)
      p[8]     = weight of kernel 2
      p[9]     = shared log-width (log_sigma) for both kernels
    
    Using exp() for width ensures smooth, positive sigma.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    bias = params[0]
    cx1, cy1, cz1 = params[1], params[2], params[3]
    w1 = params[4]
    cx2, cy2, cz2 = params[5], params[6], params[7]
    w2 = params[8]
    log_sigma = params[9]
    
    # Smooth positive width via exp; clip to avoid extreme values
    log_sigma_clipped = np.clip(log_sigma, -10.0, 10.0)
    sigma = np.exp(log_sigma_clipped)
    
    # Squared distances
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    d1_sq = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
    
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    d2_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
    
    # RBF values with safe exponent
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    rbf1 = np.exp(-d1_sq * inv_2s2)
    rbf2 = np.exp(-d2_sq * inv_2s2)
    
    return bias + w1 * rbf1 + w2 * rbf2
# EVOLVE-BLOCK-END
