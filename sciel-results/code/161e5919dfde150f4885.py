import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0 # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses Expected Improvement (EI) for minimization, scaled by cost.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    Score = EI / cost
    """
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > (TIME_LIMIT - SAFETY_MARGIN):
        return np.zeros_like(mu)

    # Expected Improvement for minimization
    # Improvement is (f_best - mu) when mu < f_best
    # z = (f_best - mu) / sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, we want to maximize EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scaling
    # Higher score is better.
    # If cost is high, we penalize the score.
    scores = ei / cost
    
    # Handle potential numerical issues (inf/nan)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
