"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses Expected Improvement (EI) with a softened cost penalty.
    """
    # Deadline guard: if time is running out, fall back to a fast exploitation strategy
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return -mu / cost to prioritize low predicted values with low cost
        # This is a simple, fast heuristic to ensure termination
        return -mu / cost

    # Compute Expected Improvement (EI)
    # sigma is the standard deviation
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scoring
    # Baseline uses EI / cost.
    # We use EI / (cost ** 0.5) to reduce the penalty for expensive evaluations,
    # allowing the optimizer to explore higher-cost regions if the improvement is promising.
    scores = ei / (cost ** 0.5)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
