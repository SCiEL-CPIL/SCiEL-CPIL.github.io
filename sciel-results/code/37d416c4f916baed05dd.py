import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates a function using 10 Gaussian Radial Basis Functions.
    params[0:10]: amplitudes for 10 RBFs
    """
    # RBF part
    # Fixed centers
    # Using 10 centers to maximize flexibility and cover the domain densely
    centers = np.linspace(-3.0, 3.0, 10)
    # Fixed width
    # Slightly wider sigma to ensure smooth gradients for BFGS and better global trend capture
    sigma = 0.8
    
    # Compute distances
    # x shape (N,), centers shape (10,)
    # We need (N, 10) for broadcasting
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    diff = x_expanded - c_expanded
    
    # Gaussian kernel
    # Using 0.5 * (diff/sigma)^2
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum of RBFs
    # rbf_values shape (N, 10)
    # params shape (10,)
    y_rbf = rbf_values @ params
    
    return y_rbf
