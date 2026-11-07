# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (CA-LCB).
    
    Strategy:
    - Compute the Lower Confidence Bound (LCB) to balance exploitation (low mu) 
      and exploration (high sigma).
    - LCB = mu - beta * sigma. We want to minimize LCB.
    - Adjust the LCB score by dividing by the evaluation cost to prioritize
      cheaper evaluations that offer similar or better potential improvement.
    - This approach is often more robust for multi-modal functions than EI,
      as it does not suffer from the "greedy" local minima trapping as severely.
    """
    # Check deadline first
    if _check_deadline():
        # Return a neutral score if we're running out of time
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e6)
    cost = np.clip(cost, 0.5, 100.0)
    
    # Hyperparameter for LCB: beta controls the exploration/exploitation trade-off
    # A larger beta encourages more exploration.
    beta = 2.0
    
    # Calculate LCB: mu - beta * sigma
    # We want to minimize this value, so we maximize -LCB
    lcb = mu - beta * sigma
    score = -lcb / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
