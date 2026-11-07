"""Rewritten program using Standard Expected Improvement (EI) for Bayesian Optimization."""
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
    Uses Standard Expected Improvement (EI) for minimization.
    EI(x) = E[max(0, f_best - f(x))]
    For f(x) ~ N(mu, sigma^2):
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return zeros to avoid timeout, leading to arbitrary selection
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For very small sigma, EI approximates max(0, f_best - mu)
    # The formula handles this naturally as phi(z) -> 0 and Phi(z) -> 1 or 0 appropriately.
    
    # Vectorized computation
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
