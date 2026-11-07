"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balance expected improvement against evaluation cost.
    
    Uses a modified expected improvement that accounts for uncertainty and cost.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Guard against non-finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Penalize high cost: divide by cost
    # Also add a small exploration bonus for uncertain regions
    # Use a temperature-like scaling to balance exploitation vs exploration
    scores = ei / cost
    
    # Add a small exploration bonus for high uncertainty
    # This encourages exploring regions with high sigma
    exploration_bonus = sigma * 0.1
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
