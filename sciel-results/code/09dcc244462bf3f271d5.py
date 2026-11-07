"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return safe scores
    # The acquisition function is called repeatedly in a loop; we need to be fast.
    # Since we can't track elapsed time easily across calls without global state,
    # and the function must be fast, we'll rely on efficient computation.
    
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Add a small exploration term based on uncertainty
    # Use a smaller weight to keep exploitation dominant
    exploration = 0.05 * sigma / cost_safe
    
    score = ei_cost + exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
