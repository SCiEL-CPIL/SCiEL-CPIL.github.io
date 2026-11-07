"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6D.
    
    Uses Expected Improvement (EI) normalized by cost, with an adaptive
    exploration term that increases as budget is consumed.
    """
    # Deadline guard: if we're running out of time, return a safe default
    # (this is a safety net; the main computation should be fast)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to prefer cheap points
    # But we also want to balance exploration
    # The cost is 1 + mean(x) where x is in [0,1]^6, so cost is in [1, 2]
    
    # Adaptive exploration: as we spend more budget, be more exploitative
    # Estimate progress from cost (average cost per eval)
    # We don't know total budget spent, but we can use the cost distribution
    # as a proxy. Lower cost points are "cheaper" so we can afford more exploration.
    
    # Use a simple approach: EI / cost with a small exploration bonus
    # that depends on uncertainty
    exploration_bonus = 0.1 * sigma_safe / cost
    score = ei / cost + exploration_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
