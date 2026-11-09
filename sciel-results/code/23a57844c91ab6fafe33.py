"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def _safe_poi(mu, sigma, f_best):
    """Compute probability of improvement with numerical safeguards."""
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    # Clip z to avoid overflow in cdf
    z = np.clip(z, -10, 10)
    poi = norm.cdf(z)
    return poi


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: POI weighted by dynamic exploration factor, normalized by cost.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std dev predictions
    - f_best: float, best observed value so far
    - cost: array of shape (2000,), evaluation costs (1 + mean(x_scaled))
    
    Returns:
    - scores: array of shape (2000,), higher = evaluate next
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure inputs are 1D numpy arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute POI
    poi = _safe_poi(mu, sigma, f_best)
    
    # Dynamic exploration factor: normalize sigma by global max sigma
    sigma_max = np.max(sigma)
    if sigma_max > 0:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Weight POI by exploration factor: higher uncertainty gets a boost
    # Use a mild weight to balance exploitation (POI) and exploration (sigma)
    exploration_weight = 0.3
    weighted_poi = poi * (1.0 + exploration_weight * sigma_norm)
    
    # Normalize by cost to prioritize low-cost regions
    scores = weighted_poi / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
