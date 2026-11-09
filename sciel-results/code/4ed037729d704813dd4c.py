"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and cost-awareness,
    with a deadline guard to ensure we finish within the time limit.
    """
    # Deadline guard: track elapsed time and return early if needed
    t_start = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware normalization: divide by cost to prefer cheaper points
    # Add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score: EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t_start > 290:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
