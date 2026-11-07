import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid strategy: UCB-like exploration weighted by cost efficiency,
    with adaptive beta that increases as uncertainty shrinks.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        safe_cost = np.maximum(cost, 1e-6)
        scores = -mu / safe_cost
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-8)

    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)

    # Compute the spread of sigma to adapt beta
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    
    # If sigma is very small, we're in exploitation phase; otherwise exploration
    # Use a logarithmic scaling for beta to be more responsive
    if sigma_max > 1e-6:
        # Beta ranges from ~0.5 (high uncertainty) to ~3.0 (low uncertainty)
        # This encourages more exploration when we're uncertain
        sigma_ratio = (sigma_max - sigma) / (sigma_max - sigma_min + 1e-12)
        beta = 0.5 + 2.5 * sigma_ratio
    else:
        beta = 2.0

    # Cost-aware LCB: we want to minimize mu - beta*sigma, but penalize by cost
    # Score = (beta * sigma - mu) / cost
    # This is the expected improvement per unit cost
    
    # Additionally, add a small exploration bonus for points with high sigma
    # relative to the global sigma, to encourage broader search
    exploration_bonus = 0.1 * (sigma / (sigma_max + 1e-12))
    
    scores = (beta * sigma - mu + exploration_bonus * sigma_max) / safe_cost

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
