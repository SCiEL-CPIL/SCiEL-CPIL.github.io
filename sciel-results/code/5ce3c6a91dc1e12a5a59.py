import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    Returns scores where higher is better (i.e., higher expected improvement).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT_LIMIT - _SAFETY_MARGIN:
        # Fallback to greedy exploitation if running out of time
        # Return improvement over f_best, which is high if mu is low
        return f_best - mu

    # Ensure sigma is non-negative and finite to avoid division by zero or NaNs
    # Clip sigma to a small positive value
    sigma = np.clip(sigma, 1e-10, None)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive, indicating improvement
    z = (f_best - mu) / sigma
    
    # Calculate standard normal PDF and CDF
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
