"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # UCB (Upper Confidence Bound) for minimization
    # We want to minimize f(x).
    # The lower bound of the confidence interval is mu - beta * sigma.
    # We select the point with the lowest lower bound.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we return the negative of the lower bound: -(mu - beta * sigma) = beta * sigma - mu.
    
    # Beta controls the exploration-exploitation trade-off.
    # A value of 2.0 is a common heuristic for moderate exploration.
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
