import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) approximation with fixed centers.
    Uses 9 RBF weights and 1 shared log-width parameter.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    weights = params[:9]
    log_sigma = params[9]
    
    # Ensure sigma is positive and reasonably bounded
    log_sigma_clipped = np.clip(log_sigma, -10.0, 10.0)
    sigma = np.exp(log_sigma_clipped)
    
    # Define 9 fixed centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Compute RBF values (Gaussian)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Exponent: -0.5 * (diff / sigma)^2
    exponent = -0.5 * (diff * diff) / (sigma * sigma)
    
    rbf_values = np.exp(exponent)
    
    # Linear combination
    prediction = rbf_values @ weights
    
    return prediction
