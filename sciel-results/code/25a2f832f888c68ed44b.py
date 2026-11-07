import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: exploit
        return -mu.copy()
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Higher EI = more expected improvement = better to evaluate
    
    # Handle sigma near zero to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Higher EI means we should evaluate this point next
    return ei
