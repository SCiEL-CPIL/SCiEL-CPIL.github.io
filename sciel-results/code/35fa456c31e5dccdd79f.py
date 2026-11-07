import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) Acquisition Function for Minimization.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Fallback to greedy (maximize -mu)
        return -mu
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e9, neginf=-1e9)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.maximum(sigma, 1e-10) # Avoid division by zero
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = E[max(f_best - Y, 0)] where Y ~ N(mu, sigma)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu < 0, z < 0.
    # The formula holds.
    
    # Optimization: For large negative z, EI is approx 0.
    # For large positive z, EI is approx f_best - mu.
    
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    EI = np.nan_to_num(EI, nan=0.0, posinf=1e9, neginf=0.0)
    
    return EI
