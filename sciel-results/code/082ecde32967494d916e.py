"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network.
    
    params[0:5]  : centers of the 5 Gaussian kernels
    params[5:10] : amplitudes of the 5 Gaussian kernels
    
    The width of each Gaussian is fixed at 0.5 to keep the optimization
    landscape smoother and more stable for BFGS.
    """
    # Extract centers and amplitudes
    centers = params[0:5]
    amplitudes = params[5:10]
    
    # Reshape for broadcasting: x is (N,), centers is (5,)
    # We want to compute (x[:, None] - centers[None, :])
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    
    # Compute squared distances: (N, 5)
    diff = x_col - c_row
    sq_dist = diff * diff
    
    # Fixed width sigma = 0.5, so 2 * sigma^2 = 0.5
    # exp(-sq_dist / (2 * sigma^2)) = exp(-sq_dist / 0.5) = exp(-2 * sq_dist)
    kernels = np.exp(-2.0 * sq_dist)
    
    # Weighted sum: (N, 5) * (5,) -> (N,)
    prediction = kernels @ amplitudes
    
    return prediction
# EVOLVE-BLOCK-END
