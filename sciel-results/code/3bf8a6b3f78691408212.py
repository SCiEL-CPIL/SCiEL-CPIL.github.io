"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-style local basis function expansion.
    
    Uses 10 fixed Gaussian kernels centered evenly across the domain [0.1, 6.0].
    The 10 parameters represent the weights for each kernel.
    """
    # 1. Define fixed centers evenly spaced in [0.1, 6.0]
    # 10 points, so spacing is (6.0 - 0.1) / 9
    centers = np.linspace(0.1, 6.0, 10)
    
    # 2. Fixed bandwidth (sigma)
    # With 10 points over range ~5.9, spacing is ~0.66.
    # Sigma = 0.6 allows significant overlap for smoothness while maintaining local sensitivity.
    sigma = 0.6
    
    # 3. Compute kernel values
    # x shape: (n,)
    # centers shape: (10,)
    # diff shape: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Squared exponential kernel: exp(-diff^2 / (2 * sigma^2))
    # (n, 10)
    kernel_matrix = np.exp(-diff ** 2 / (2.0 * sigma ** 2))
    
    # 4. Prediction: weighted sum of kernels
    # params shape: (10,)
    # kernel_matrix @ params -> (n,)
    pred = kernel_matrix @ params
    
    return pred
# EVOLVE-BLOCK-END
