"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Compute expected improvement for points where mu < f_best."""
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Clip to non-negative
    ei = np.maximum(ei, 0.0)
    return ei


def _ucb(mu, sigma, f_best, xi=0.0, beta=2.0):
    """Upper confidence bound (for minimization, use lower bound)."""
    return mu - beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that balances improvement and cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more desirable)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return a reasonable default
        return np.zeros_like(mu)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Compute EI
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Compute a cost-aware score.
    # We want to favor points with high EI relative to their cost.
    # Use a soft normalization to avoid division by very small costs.
    
    # Strategy: combine EI with a penalty for high cost.
    # The cost is in [1, 2] (since 1 + mean(x) where x in [0,1]^4).
    # We can use: score = EI / cost^alpha, with alpha ~ 1.0-1.5
    
    # Let's try a more nuanced approach:
    # 1. Normalize EI to [0, 1] range for this batch
    # 2. Normalize cost to [0, 1] range
    # 3. Combine them
    
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Combine: favor high EI and low cost
    # score = ei_norm * (1 - cost_norm * gamma)
    # where gamma controls the strength of cost penalty
    
    gamma = 0.5  # Moderate cost penalty
    scores = ei_norm * (1.0 - gamma * cost_norm)
    
    # Also add a small exploration bonus for uncertain points
    # to avoid getting stuck in local optima
    sigma_max = np.max(sigma) if np.max(sigma) > 0 else 1.0
    sigma_norm = sigma / sigma_max
    
    # Small exploration term
    exploration_weight = 0.1
    scores += exploration_weight * sigma_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
