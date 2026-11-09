"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290  # 290s safety margin from 300s limit
    return _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using EI with exponential cost penalty.
    
    Instead of dividing by linear cost, we multiply EI by exp(-alpha * cost)
    to exponentially suppress high-cost regions. This better aligns with the
    Levy 2D landscape where the minimum is near the center (low cost).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Safety: handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Exponential cost penalty: strongly suppresses high-cost regions
    # alpha controls the strength of cost aversion
    alpha = 2.0
    cost_penalty = np.exp(-alpha * cost)
    
    # Combined acquisition: EI * exponential cost penalty
    scores = ei * cost_penalty
    
    # Add small UCB term also weighted by cost penalty for exploration
    beta = 0.5
    ucb_bonus = beta * sigma * cost_penalty
    
    scores = scores + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
