"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Cost-Weighted Probability of Improvement (PI) with a dynamic
    exploration floor based on normalized sigma. This favors cheap,
    high-probability improvements while maintaining exploration in
    uncertain regions.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x) where x is in [0,1]^3.
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Check deadline: if we're running out of time, return a simple fallback
    if _time_left() <= 0:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure sigma and cost are positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for Probability of Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Probability of Improvement (PI) for minimization: Phi(z)
    pi = norm.cdf(z)
    
    # Dynamic exploration bonus: normalized sigma acts as uncertainty bonus
    # that decays as the GP becomes more confident
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        exploration_bonus = 0.1 * (sigma_safe / sigma_max)
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    # Combined score: PI + exploration bonus
    combined = pi + exploration_bonus
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
