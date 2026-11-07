"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a linear cost penalty.
    """
    # Time guard
    t_start = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using log-sum-exp or careful calculation for stability if needed, 
    # but standard formula is usually fine for 2000 points.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0)
    
    # Cost-aware score: divide by cost to prefer cheaper evaluations
    # This fits more evaluations within the budget
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard check
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        # Return a simple fallback (e.g., just EI without cost if cost is problematic, or uniform)
        # But cost is always >= 1, so it's safe.
        return ei
    
    return score
# EVOLVE-BLOCK-END
