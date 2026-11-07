"""Rewritten program: RBF basis with fixed centers and shared learnable width."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 RBFs at fixed, evenly spaced centers.
    
    params:
        params[0]: shared width parameter (sigma)
        params[1:11]: coefficients for the 10 RBF centers
    
    Centers are fixed at evenly spaced points in [-3, 3].
    We use a softplus-like transform on params[0] to keep width positive,
    or simply use abs(params[0]) + small epsilon to ensure positivity.
    """
    # Fixed centers evenly spaced in [-3, 3]
    # 10 centers: from -3 to 3, step = 6/9
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Shared width: use abs(params[0]) + 0.1 to ensure positive and avoid zero
    sigma = np.abs(params[0]) + 0.1
    
    # x is 1D array of shape (n,)
    x = np.asarray(x)
    
    # Compute RBF values: shape (n, 10)
    # For each sample i and center j: exp(-((x_i - center_j)^2) / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    rbf_values = np.exp(-0.5 * (diff ** 2) / (sigma ** 2))  # (n, 10)
    
    # Coefficients
    coeffs = params[1:11]  # shape (10,)
    
    # Prediction: (n, 10) @ (10,) -> (n,)
    prediction = rbf_values @ coeffs
    
    return prediction
# EVOLVE-BLOCK-END
