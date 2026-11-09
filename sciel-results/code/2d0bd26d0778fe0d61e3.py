"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balances expected improvement against evaluation cost.
    
    Uses a hybrid of expected improvement (EI) and probability of improvement (PI)
    with cost normalization, plus a small exploration bonus for low-cost regions.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Probability of Improvement (PI) - captures the chance of finding a better point
    pi = norm.cdf(z)
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    # cost = 1 + mean(x_scaled), so cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    
    # Blend EI and PI: EI is more aggressive, PI is more conservative
    # Weight them to balance exploitation and exploration
    blended = 0.7 * ei + 0.3 * pi * sigma_safe
    
    # Cost-aware normalization
    score = blended / cost_safe
    
    # Add a small exploration bonus for points with high uncertainty (high sigma)
    # but only if they're not too expensive
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    total_score = score + exploration_bonus
    
    # Ensure finite values
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    # Ensure non-negative scores (higher is better)
    total_score = np.maximum(total_score, 0.0)
    
    return total_score
# EVOLVE-BLOCK-END
