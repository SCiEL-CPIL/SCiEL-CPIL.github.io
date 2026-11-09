"""Cost-aware acquisition for Ackley 2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Cost-Weighted Upper Confidence Bound (UCB) as the primary signal.
    This structural change from EI aims to improve exploration, which has been
    observed to yield better results in similar contexts.
    
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
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # UCB Parameter: Beta
    # Beta controls the exploration/exploitation trade-off.
    # A value of 2.0 is often effective for standardized GPs in small-budget BO.
    beta = 2.0
    
    # Compute UCB Score
    # For minimization: we want low mu and high sigma.
    # Score = (f_best - mu) + beta * sigma
    # This is positive if the potential improvement plus uncertainty exceeds the current best.
    ucb_scores = (f_best - mu) + beta * sigma
    
    # Cost-aware score
    # We divide by cost to prioritize cheaper points that offer similar "value".
    # Cost is at least 1.0 based on the problem description (1 + mean(x_scaled)).
    scores = ucb_scores / cost
    
    # Replace any remaining NaN/Inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
