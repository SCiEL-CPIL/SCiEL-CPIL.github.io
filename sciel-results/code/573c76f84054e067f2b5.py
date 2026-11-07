"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (POI) for minimization.
    
    Strategy:
    - Use POI: score = P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma)
    - This provides a binary-like exploration signal that differs from UCB.
    - Candidates with high probability of improving over f_best get high scores.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # POI for minimization: P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma)
    # Higher values mean higher probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with very large values
    z_clipped = np.clip(z, -10, 10)
    
    scores = norm.cdf(z_clipped)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
