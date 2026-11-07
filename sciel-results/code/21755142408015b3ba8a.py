"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 5 kernels.
    
    First 5 params: centers c_i for each Gaussian kernel
    Last 5 params: amplitudes a_i for each Gaussian kernel
    
    f(x) = sum_{i=1}^{5} a_i * exp(-((x - c_i)^2) / (2 * sigma^2))
    
    The fixed width sigma is chosen to be ~0.8, giving moderate locality.
    This local basis can capture sharp features and discontinuities that
    global polynomial bases struggle with.
    """
    centers = params[:5]
    amplitudes = params[5:]
    
    # Fixed width for numerical stability
    sigma = 0.8
    
    # Compute RBF values: shape (n_points, 5)
    # x shape: (n_points,), centers shape: (5,)
    # diff shape: (n_points, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination
    result = rbf_values @ amplitudes
    
    return result
# EVOLVE-BLOCK-END
