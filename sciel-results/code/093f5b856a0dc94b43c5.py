# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute centers for RBF
# 10 centers uniformly spaced in [0.1, 6]
_CENTERS = np.linspace(0.1, 6.0, 10)
# Fixed width for Gaussian kernels
# Spacing is ~0.65, so sigma ~ 0.8 provides good overlap
_SIGMA = 0.8
_SIGMA2_2 = 2.0 * _SIGMA * _SIGMA

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) network with 10 Gaussian kernels.
    Fixed centers, fixed width, learnable weights.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Compute distances from x to each center
    # x shape: (N,) or (N, 1)
    # We want a matrix of shape (10, N) for basis
    x_flat = x.flatten()
    
    # Vectorized computation of Gaussian kernels
    # diff[i, j] = x[j] - _CENTERS[i]
    diff = x_flat[np.newaxis, :] - _CENTERS[:, np.newaxis]
    
    # Kernel values
    basis = np.exp(-diff**2 / _SIGMA2_2)
    
    # Linear combination
    y = p @ basis
    
    return y
# EVOLVE-BLOCK-END
