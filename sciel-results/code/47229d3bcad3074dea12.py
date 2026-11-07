import numpy as np
from scipy.special import erf
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Lower Confidence Bound (LCB) for minimization, normalized by cost.
    LCB = mu - beta * sigma. We want to minimize LCB, so we maximize -LCB / cost.
    Score = (beta * sigma - mu) / cost.
    """
    # Deadline guard: If we are running out of time, switch to greedy exploitation
    if time.time() > _DEADLINE:
        safe_cost = np.maximum(cost, 1e-6)
        scores = -mu / safe_cost
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-6)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # LCB Parameter
    # Beta controls exploration. 2.0 is a standard choice.
    beta = 2.0
    
    # Calculate Score: (beta * sigma - mu) / cost
    # This is equivalent to minimizing (mu - beta * sigma) * cost
    scores = (beta * sigma - mu) / safe_cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
