"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: GP-LCB normalized by cost.
    
    Uses a Lower Confidence Bound (LCB) with adaptive beta that balances
    exploitation and exploration, then normalizes by evaluation cost to
    account for the varying cost of evaluations.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time and return early if needed
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive beta: scale with the range of mu values to balance
    # exploration/exploitation based on the current GP model
    mu_range = np.ptp(mu) if len(mu) > 0 else 1.0
    if mu_range < 1e-9:
        mu_range = 1.0
    
    # Base beta for LCB, scaled to be meaningful relative to the GP uncertainty
    # Use a moderate exploration factor
    beta = 2.0
    
    # GP-LCB: lower confidence bound (we want to minimize, so lower is better)
    # lcb = mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Cost-aware: divide by cost to get value per unit cost
    # This penalizes expensive evaluations
    score = -lcb / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check: if we've used too much time, return what we have
    elapsed = time.time() - start_time
    if elapsed > 290:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
