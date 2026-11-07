import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement with exploration parameter xi
    xi = 0.1
    
    # Handle division by zero or very small sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Target improvement
    target = f_best - xi
    
    # Z-score
    z = (target - mu) / sigma_safe
    
    # EI calculation
    ei = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative values to 0
    ei = np.maximum(ei, 0)
    
    return ei
