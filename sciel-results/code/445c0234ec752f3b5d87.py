"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Uses Expected Improvement (EI) normalized by cost, with a small
    exploration bonus to balance exploitation and exploration.
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    if _is_deadline_exceeded():
        # Return scores that prioritize low cost and low predicted mean
        return -(mu + 0.1 * cost)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost (higher cost = lower score)
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps avoid premature convergence
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    # Combine exploitation (EI) and exploration
    final_score = score + exploration_bonus
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
