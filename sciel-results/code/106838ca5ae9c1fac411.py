import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # To avoid overflow in exp for very negative z, we can use stable forms, 
    # but scipy's norm.cdf and norm.pdf are quite stable.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
