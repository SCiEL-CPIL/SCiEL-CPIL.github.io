"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed-width RBF model with 10 centers evenly spaced across [-3, 3]
    # The 10 parameters are the RBF weights (linear model in params)
    # This ensures BFGS converges to global optimum for the linear problem
    
    # Centers evenly spaced: -3, -2.333, ..., 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Moderate global width (sigma)
    sigma = 0.5
    sigma2 = sigma ** 2
    
    # Compute RBF values: exp(-0.5 * (x - c)^2 / sigma^2)
    # x shape: (N,), centers shape: (10,)
    # We want output shape (N,)
    
    # Efficient computation using broadcasting
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF values shape: (N, 10)
    rbf_values = np.exp(-0.5 * diff ** 2 / sigma2)
    
    # params shape: (10,)
    # result = rbf_values @ params
    result = rbf_values @ params
    
    return result
# EVOLVE-BLOCK-END
