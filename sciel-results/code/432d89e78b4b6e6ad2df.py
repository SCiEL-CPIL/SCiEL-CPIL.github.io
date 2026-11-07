import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a Gaussian Process Regression (Kernel) formulation.
    
    y(x) = sum_{i=0}^{9} w_i * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    Centers c_i are evenly spaced in [0.1, 6].
    Sigma is fixed to provide local flexibility.
    Weights w_i are the 10 parameters.
    
    This is a linear model in the parameters, so BFGS converges reliably.
    """
    x = np.asarray(x, dtype=float)
    
    # Define RBF centers evenly spaced in the domain [0.1, 6]
    n_centers = 10
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Define width (sigma) for the Gaussian kernels
    # A width of roughly half the spacing between centers ensures good overlap
    # Spacing is (6.0 - 0.1) / 9 ~ 0.65. Sigma = 0.5 is a reasonable choice.
    sigma = 0.5
    sigma2 = 2.0 * sigma * sigma
    
    # Compute the kernel matrix: K[j, i] = exp(-(x_j - c_i)^2 / sigma2)
    # x has shape (n_points,), centers has shape (n_centers,)
    x_expanded = x[:, np.newaxis]  # (n_points, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, n_centers)
    
    diff = x_expanded - centers_expanded  # (n_points, n_centers)
    K = np.exp(-diff * diff / sigma2)  # (n_points, n_centers)
    
    # y = K @ w, where w = params
    y = K @ params
    
    return y
