"""RBF network for symbolic regression with learnable centers and amplitudes."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 5 kernels.
    params[0:5] are the centers c_i in [-2, 2]
    params[5:10] are the amplitudes a_i
    Fixed width sigma = 0.5 for stability
    """
    centers = params[0:5]
    amplitudes = params[5:10]
    
    # Fixed width for numerical stability
    sigma = 0.5
    
    # Compute Gaussian kernels: exp(-0.5 * ((x - c_i) / sigma)^2)
    # Vectorized computation
    # x shape: (N,)
    # centers shape: (5,)
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute squared distances normalized by sigma^2
    sq_dist = (diff ** 2) / (2.0 * sigma ** 2)
    
    # Apply exponential
    kernels = np.exp(-sq_dist)
    
    # Linear combination: y = sum(a_i * kernel_i(x))
    # kernels shape: (N, 5)
    # amplitudes shape: (5,)
    result = kernels @ amplitudes
    
    return result
# EVOLVE-BLOCK-END
