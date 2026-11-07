"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    UCB-based cost-aware acquisition function.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time to avoid timeout
    t_start = time.time()
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # UCB acquisition: mu + beta * sigma
    # For minimization, we want low mu and high sigma (exploration)
    # UCB for minimization: mu - beta * sigma (we want to minimize, so explore where sigma is high)
    # Actually, for minimization, the "upper confidence bound" in the sense of
    # "how low could this be" is mu - beta * sigma
    # But the standard UCB for maximization is mu + beta * sigma
    # Since we're minimizing, we can think of it as maximizing the negative:
    # Let g(x) = -f(x), then we maximize g. GP on g has mean -mu, std sigma.
    # UCB for g: -mu + beta * sigma
    # So acquisition = -mu + beta * sigma
    
    beta = 2.0
    
    # Compute UCB score (for the transformed maximization problem)
    ucb = -mu + beta * sigma
    
    # Cost-aware: divide by cost to balance exploration/exploitation with cost
    # Use linear cost penalty
    scores = ucb / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline check: if we've used too much time, return immediately
    elapsed = time.time() - t_start
    if elapsed > 280.0:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
