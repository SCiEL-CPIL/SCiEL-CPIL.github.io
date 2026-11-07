import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    # Higher EI indicates higher expected improvement over f_best
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    Z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    
    return ei
