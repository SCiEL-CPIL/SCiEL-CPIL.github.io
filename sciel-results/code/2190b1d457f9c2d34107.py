"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer to ensure we don't exceed the 300s limit if called repeatedly or in a tight loop
# Although 'acquisition' is a stateless function, we can use a module-level variable
# to track time if we decide to add complex logic later. For now, it's a safety net.
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin of 10s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses UCL (Upper Confidence Limit) for minimization, normalized by cost.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviation predictions
    f_best: scalar, best observed value so far
    cost: (2000,) array of costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    # Deadline guard: If we are close to the limit, return a simple score to ensure we don't hang
    # This is primarily a safeguard against infinite loops or extremely slow numpy ops (unlikely here).
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple inverse cost
        return 1.0 / (cost + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCL for minimization:
    # We want to minimize the function. The UCL lower bound is mu - kappa * sigma.
    # We want to maximize the score, so we use: Score = (kappa * sigma - mu) / cost
    # This favors high uncertainty (sigma) and low mean (mu).
    
    # Hyperparameter kappa: 2.5 is a common choice for exploration in multimodal functions
    kappa = 2.5
    
    # Calculate UCL score
    ucl_score = (kappa * sigma - mu) / cost_safe
    
    # Ensure finiteness
    ucl_score = np.where(np.isfinite(ucl_score), ucl_score, -np.inf)
    
    return ucl_score
# EVOLVE-BLOCK-END
