# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        # Fallback to exploitation if out of time
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # We want to maximize EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF of standard normal
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
