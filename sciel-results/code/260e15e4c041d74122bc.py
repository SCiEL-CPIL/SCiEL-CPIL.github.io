"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters basis of 10 Gaussian Radial Basis Functions (RBFs).
    Centers are evenly spaced across [-3, 3].
    params[0..9] are coefficients for each Gaussian.
    
    The Gaussian width is fixed to provide a good balance between
    locality and smoothness for the given domain.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 10 evenly spaced centers in [-3, 3]
    # np.linspace(-3, 3, 10) gives: -3, -2.333, -1.667, -1, -0.333, 0.333, 1, 1.667, 2.333, 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Gaussian width: fixed for stability
    # A width of about 1.0 is reasonable for a domain of length 6 with 10 centers
    # sigma = 1.0
    sigma = 1.0
    sigma_sq_2 = 2.0 * sigma * sigma  # 2 * sigma^2 = 2.0
    
    # Compute differences: shape (n, 10)
    # x is (n,), centers is (10,)
    # diff[i, j] = x[i] - centers[j]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian basis values: shape (n, 10)
    # phi[i, j] = exp(-diff[i, j]^2 / (2 * sigma^2))
    phi = np.exp(-diff * diff / sigma_sq_2)
    
    # Compute prediction: params @ phi
    # params is (10,), phi is (n, 10)
    pred = phi @ params
    
    return pred
# EVOLVE-BLOCK-END
