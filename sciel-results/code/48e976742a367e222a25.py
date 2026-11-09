import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    EI is a robust choice for Bayesian optimization that balances exploration
    and exploitation based on the posterior uncertainty.
    """
    # Track time to ensure we finish well within limits
    start_time = time.time()
    timeout = 280  # Safety margin from 300s limit
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use vectorized operations for speed
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Check time
    if time.time() - start_time > timeout:
        # Fallback to a simple score if taking too long (unlikely for 2000 points)
        # But just in case, return a valid array
        return np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
