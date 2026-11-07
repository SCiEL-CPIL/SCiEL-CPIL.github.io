# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 RBFs at fixed, evenly spaced centers.
    params[0] is the shared width (sigma) for all kernels.
    params[1:11] are the weights for each of the 10 kernels.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Extract width and weights
    # Use a softplus-like transformation or just abs to keep width positive and avoid division by zero
    # To ensure smoothness and differentiability for BFGS, we use a small epsilon offset
    # sigma = params[0]
    # If params[0] is very small, the RBFs become spikes, which can cause numerical issues.
    # Let's enforce a minimum width.
    sigma = np.abs(params[0]) + 1e-3
    
    # Define 10 fixed, evenly spaced centers in [-3, 3]
    # np.linspace(-3, 3, 10) gives centers: -3, -2.333, ..., 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Compute distances: (N, 1) - (1, 10) -> (N, 10)
    # x is (N,), centers is (10,)
    dist2 = (x[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    
    # Compute RBF values: exp(-dist2 / (2 * sigma^2))
    rbf_values = np.exp(-dist2 / (2.0 * sigma ** 2))
    
    # Weights are params[1:11]
    weights = params[1:11]
    
    # Linear combination
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
