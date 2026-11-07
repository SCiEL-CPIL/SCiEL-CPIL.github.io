"""Symbolic regression with 1 input; Linear-in-parameters Gaussian RBF model."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of Gaussian Radial Basis Functions (RBFs).
    
    Model:
    y(x) = sum_{i=1}^{8} w_i * exp(-(x - c_i)^2 / (2 * sigma^2)) + m * x + b
    
    Parameters (10 total):
    - p[0:8]: Weights for the 8 Gaussian RBFs
    - p[8]: Linear coefficient for x
    - p[9]: Bias term
    
    Fixed Hyperparameters:
    - Centers c_i: Evenly spaced in [0.5, 5.5]
    - Width sigma: 1.0
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    # p[0:8] are RBF weights
    w = params[:8]
    # p[8] is linear slope
    m = params[8]
    # p[9] is bias
    b = params[9]
    
    # Fixed centers covering the domain [0.1, 6]
    # We use 8 centers. Spacing is roughly (5.5 - 0.5) / 7 = 0.714
    centers = np.array([0.5, 1.214, 1.928, 2.642, 3.357, 4.071, 4.785, 5.5])
    
    # Fixed width for all Gaussians
    sigma = 1.0
    sigma2 = 2.0 * sigma * sigma
    
    # Compute Gaussian basis values
    # Shape: (N, 8)
    # diff = x[:, None] - centers[None, :]
    diff = x[:, None] - centers[None, :]
    
    # Compute exponent part
    # exp(-diff^2 / sigma2)
    basis = np.exp(-diff * diff / sigma2)
    
    # Linear combination of basis functions
    # y_rbf = basis @ w
    y_rbf = basis @ w
    
    # Add linear and bias terms
    y = y_rbf + m * x + b
    
    return y
# EVOLVE-BLOCK-END
