import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback to pure exploitation if time is tight
        return -mu.copy()
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Clip to 0 to ensure non-negative scores
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
