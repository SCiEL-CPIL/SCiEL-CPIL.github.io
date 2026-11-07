import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Gaussian Radial Basis Functions (RBFs).
    Uses 10 fixed centers evenly distributed in [-2, 2] and a fixed width.
    The model is linear in the parameters, ensuring stable optimization.
    """
    # Define fixed centers for the RBFs
    centers = np.array([-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8])
    
    # Define the width (sigma) for the Gaussian kernels
    sigma = 0.5
    
    # Compute the RBF values
    # x shape: (N,)
    # centers shape: (10,)
    # We need (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-(x - c)^2 / (2 * sigma^2))
    rbf_values = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Linear combination
    # params shape: (10,)
    # rbf_values shape: (N, 10)
    return rbf_values.dot(params)
