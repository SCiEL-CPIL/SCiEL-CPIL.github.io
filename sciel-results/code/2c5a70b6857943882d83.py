"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    mu, sigma: shape (2000,), GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: shape (2000,), evaluation cost for each candidate
    """
    # Time guard: if we've been running too long, return a safe fallback
    # The caller manages the overall loop; we just ensure we don't hang.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # cost is 1 + mean(x) where x in [0,1]^3, so cost in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
