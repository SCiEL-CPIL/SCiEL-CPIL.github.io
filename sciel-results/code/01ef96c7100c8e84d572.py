import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle division by zero in sigma
    safe_sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    
    z = (f_best - mu) / safe_sigma
    
    # Calculate EI
    # If sigma is very small, z is large.
    # If mu < f_best, z is positive, EI is positive.
    # If mu > f_best, z is negative, EI is 0 (theoretically, numerically small).
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
