"""Cost-aware acquisition for Ackley 2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement per unit cost (EI/cost) as the primary signal,
    with a floor on cost to prevent division issues, and handles edge cases.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: uniform-ish scores based on low mu
        # This should rarely be hit, but we need to return valid finite values
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        # Fallback: use -mu/cost as a simple proxy
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return fallback_scores
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace NaN/Inf with reasonable defaults
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute z-score for EI/UCB
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Hybrid Acquisition: Combine Exploitation (f_best - mu) and Exploration (sigma)
    # This is similar to UCB: mu - beta * sigma, but we want to maximize score for minimization.
    # Score = (f_best - mu + beta * sigma) / cost
    # beta controls exploration. 1.0 is a reasonable default.
    beta = 1.0
    raw_scores = (f_best - mu) + beta * sigma
    
    # Cost-aware: divide by cost
    # The task says cost = 1 + mean(x scaled to [0,1]), so cost is in [1, 2]
    # Higher cost means we want to penalize those points more
    scores = raw_scores / cost
    
    # Replace any remaining NaN/Inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
