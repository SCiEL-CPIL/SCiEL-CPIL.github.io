"""Cost-aware acquisition for Bayesian optimization on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining expected improvement
    with an exploration term, normalized by cost.
    
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
        Acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, _EPS)
    
    # Standardize for numerical stability
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage exploring high-uncertainty regions
    # Scale by a factor relative to the range of mu
    mu_range = np.ptp(mu) if len(mu) > 1 else 1.0
    if mu_range < _EPS:
        mu_range = 1.0
    
    # Exploration term: sigma normalized, with a small weight
    exploration_weight = 0.05
    exploration = exploration_weight * sigma_safe / mu_range
    
    # Combined score: EI + exploration, divided by cost
    # Add small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, _EPS)
    
    scores = (ei + exploration) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
