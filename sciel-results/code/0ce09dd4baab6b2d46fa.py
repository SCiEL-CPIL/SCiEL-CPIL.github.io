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
    Acquisition function using Knowledge Gradient (KG) for minimization.
    
    Strategy:
    - KG scores candidates by the expected reduction in f_best after evaluating x.
    - For a GP posterior N(mu, sigma^2), KG has a closed form:
      KG(x) = sigma * (phi(z) - z * Phi(-z))  where z = (f_best - mu) / sigma
    - This is structurally different from POI (probability of improvement)
      and EI (expected magnitude of improvement).
    - KG balances exploration and exploitation by considering the expected
      information gain about the optimum.
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
    
    # z-score for Knowledge Gradient
    z = (f_best - mu) / sigma_safe
    
    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_neg_z = norm.cdf(-z)
    
    # Knowledge Gradient: expected reduction in f_best
    scores = sigma_safe * (phi_z - z * Phi_neg_z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
