"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses a Cost-Adjusted Expected Improvement (EI) strategy with modified cost
normalization to balance exploration and exploitation while respecting budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

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
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure cost is within expected range [1, 2] to avoid division by zero or extreme values
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (should be by definition, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Modified cost normalization: divide by (cost - 0.5) instead of cost
    # This amplifies the cost-awareness signal, steering search more effectively
    # towards low-cost regions where the GP might have less uncertainty or where
    # the function is easier to minimize
    cost_adjusted = cost - 0.5
    cost_adjusted = np.maximum(cost_adjusted, 0.1)  # Avoid division by zero or very small values
    
    # Cost-aware score
    score = ei / cost_adjusted
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
