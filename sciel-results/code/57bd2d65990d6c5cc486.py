"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per unit cost,
    with a small exploration bonus for low-cost regions and high uncertainty.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of shape (2000,) with finite scores (higher = evaluate next)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: cost is 1 + mean(x scaled to [0,1]), so cost in [1, 2]
    # We want to prefer lower cost, so divide by cost
    # But we also want to add a small exploration bonus for high uncertainty
    # that isn't fully captured by EI (e.g., when EI is near zero but sigma is high)
    
    # Cost-aware EI
    ei_per_cost = ei / np.maximum(cost, 1e-10)
    
    # Add a small uncertainty bonus: prefer points with high sigma when EI is low
    # This helps exploration in early stages
    # The bonus is scaled down by cost to still prefer cheaper evaluations
    sigma_bonus = sigma_safe / np.maximum(cost, 1e-10) * 0.01
    
    # Combine
    score = ei_per_cost + sigma_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
