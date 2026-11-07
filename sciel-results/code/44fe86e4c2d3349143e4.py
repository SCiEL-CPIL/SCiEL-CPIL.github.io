"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Design: 10 Gaussian kernels with fixed evenly spaced centers
    # params[0:10] : weights for the 10 kernels
    
    # Fixed centers evenly spaced across [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Weights for kernels from params
    weights = params
    
    # Fixed bandwidth - chosen to provide local smoothing
    # Distance between centers is ~0.65. Sigma ~ 0.5-0.7 seems reasonable.
    sigma = 0.7
    
    # x shape: (n,)
    # centers shape: (10,)
    # Compute distances: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Squared exponential kernel: exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff ** 2 / (2.0 * sigma ** 2))
    
    # Prediction: weighted sum of kernels
    # (n, 10) @ (10,) -> (n,)
    pred = kernel @ weights
    
    return pred
# EVOLVE-BLOCK-END
