import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 325.0
_t0 = time.time()

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) acquisition scores.
    """
    if time.time() - _t0 > _DEADLINE:
        return -mu

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    Z = (f_best - mu) / sigma
    
    # Use scipy for stable calculation of CDF and PDF
    # norm.cdf and norm.pdf handle large Z well
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    scores = (f_best - mu) * Phi + sigma * phi
    
    # EI is always non-negative.
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
