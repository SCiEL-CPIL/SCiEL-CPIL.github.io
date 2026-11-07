"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Kernel Ridge Regression strategy with Gaussian RBF kernels.
    # We use 10 fixed, evenly spaced centers in [-3, 3].
    # The 10 parameters act as weights for these kernels.
    # 
    # Centers: -3, -2.333, ..., 3
    # Width (bandwidth) is fixed to a reasonable value for local interpolation.
    # A bandwidth of ~0.8 allows each kernel to cover a local region while
    # still having some overlap for smoothness.
    
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed bandwidth (inverse of 2*sigma^2 in the exponent)
    # Let sigma = 0.5, so gamma = 1/(2*sigma^2) = 1/0.5 = 2.0
    # Narrower kernels reduce collinearity and may help BFGS converge better
    # for functions with local features.
    gamma = 2.0
    
    # Compute distances from each input point to each center
    # x shape: (n,), centers shape: (10,)
    # diff shape: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian RBF values
    # rbf shape: (n, 10)
    rbf = np.exp(-gamma * diff ** 2)
    
    # Linear combination with params (10,)
    # prediction shape: (n,)
    prediction = rbf @ params
    
    return prediction
# EVOLVE-BLOCK-END
