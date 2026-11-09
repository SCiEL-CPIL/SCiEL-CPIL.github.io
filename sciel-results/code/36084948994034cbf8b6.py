"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 5s safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with budget-adaptive exploration.
    
    Uses Expected Improvement per unit cost as the core signal, with an
    adaptive exploration bonus that decays as the cost budget is consumed.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # UCB = -mu + beta * sigma
    # We maximize this score.
    # Beta controls the trade-off between exploitation (-mu) and exploration (sigma).
    # A higher beta encourages more exploration.
    # Cost-awareness is achieved by dividing by the evaluation cost.
    
    # Use a moderate exploration parameter. 
    # 2.0 is a common starting point for standardized GPs.
    beta = 2.0
    
    ucb = -mu + beta * sigma
    
    # Cost-aware score
    score = ucb / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
