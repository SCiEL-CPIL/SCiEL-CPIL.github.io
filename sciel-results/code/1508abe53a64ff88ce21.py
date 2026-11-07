import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    # Check time deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Pure exploitation. 
        # We want to minimize f, so we want the lowest mu.
        # Score should be high for low mu.
        # Returning -mu makes lower mu result in higher score.
        return -mu.copy()

    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Ensure sigma is positive and not zero to avoid division by zero
    # A very small epsilon prevents numerical instability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute the normalized improvement
    # alpha = (f_best - mu) / sigma
    alpha = (f_best - mu) / sigma_safe
    
    # Compute Phi (CDF) and phi (PDF) of the standard normal distribution
    # norm.cdf returns Phi
    # norm.pdf returns phi
    cdf_vals = norm.cdf(alpha)
    pdf_vals = norm.pdf(alpha)
    
    # Calculate Expected Improvement
    # Note: If mu is very close to f_best and sigma is small, this handles the limit well.
    # If sigma is effectively 0 (clipped to eps), alpha will be huge if mu < f_best, 
    # cdf ~ 1, pdf ~ 0, EI ~ f_best - mu.
    # If mu > f_best, alpha is huge negative, cdf ~ 0, pdf ~ 0, EI ~ 0.
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure finite scores
    # EI should always be non-negative for valid inputs, but numerical issues can occur.
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    
    # EI is already a score where higher is better (more expected improvement).
    return ei
