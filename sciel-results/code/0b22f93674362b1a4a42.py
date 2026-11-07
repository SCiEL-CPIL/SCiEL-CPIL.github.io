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
    Acquisition function using a hybrid of Probability of Improvement (POI) 
    and an Uncertainty Bonus (from UCB).
    
    Strategy:
    - Base score is POI: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    - Add a small uncertainty bonus: beta * sigma
    - This combines the robustness of POI (which scored 1.65) with the 
      exploration capability of UCB (which scored 1.48), potentially 
      allowing escape from local plateaus where POI saturates.
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
    
    # POI for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    poi_scores = norm.cdf(z_clipped)
    
    # Uncertainty bonus (similar to UCB exploration term)
    # Using a small beta to not overwhelm the POI signal
    beta = 0.1 
    uncertainty_bonus = beta * sigma_safe
    
    # Combine: POI + Bonus
    scores = poi_scores + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
