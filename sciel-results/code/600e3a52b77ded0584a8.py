"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a Gaussian Radial Basis Function (RBF) expansion with 10 fixed centers
    to the data. Input x is in [-2, 2].
    params contains 10 amplitudes for the Gaussian kernels.
    """
    # Fixed centers evenly spaced across the domain [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for all Gaussians
    sigma = 0.5
    
    # Compute differences between each x and each center
    # x is shape (N,), centers is shape (10,)
    # diff is shape (N, 10)
    diff = x[:, None] - centers[None, :]
    
    # Compute Gaussian values: exp(-diff^2 / (2 * sigma^2))
    gaussians = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination of Gaussians with the given params
    result = gaussians @ params
    
    return result
# EVOLVE-BLOCK-END
