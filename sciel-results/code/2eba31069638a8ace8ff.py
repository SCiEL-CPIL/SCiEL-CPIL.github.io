import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: exploit best observed
        return -mu.copy()
    
    # Expected Improvement (EI) for minimization
    # EI(x) = max(0, f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # Higher EI means better candidate to evaluate next
    
    # Handle zero or very small sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Using scipy for accuracy
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But only where f_best > mu (i.e., z > 0), otherwise EI = 0
    ei = np.where(z > 0, (f_best - mu) * cdf + sigma_safe * pdf, 0.0)
    
    # Ensure finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    
    return ei
