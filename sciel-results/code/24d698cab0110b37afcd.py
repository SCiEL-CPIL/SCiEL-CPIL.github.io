"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = time.time() + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return uniform random scores if we're running out of time
    if time.time() > _DEADLINE:
        return np.random.uniform(0, 1, size=len(mu))
    
    # Cost-aware acquisition: balance exploration (low sigma) and exploitation (low mu)
    # while accounting for evaluation cost
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    score = ei / cost
    
    # Add a small exploration bonus for high uncertainty
    # This helps escape local optima in the 6D space
    exploration_bonus = 0.1 * sigma_safe / cost
    
    return score + exploration_bonus
# EVOLVE-BLOCK-END
