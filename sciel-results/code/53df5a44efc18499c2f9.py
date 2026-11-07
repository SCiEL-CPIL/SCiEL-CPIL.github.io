"""Rewritten program using Expected Improvement (EI) for Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 320s deadline, 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization.
    Uses Expected Improvement (EI) for minimization.
    EI(x) = E[max(0, f_best - f(x))]
    For f(x) ~ N(mu, sigma^2):
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return random scores or zeros to avoid timeout
        # Returning zeros will lead to arbitrary selection
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero
    # Noise is 1e-6, so sigma should be at least that, but safe to clip
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Calculate Standard Normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate Expected Improvement
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
