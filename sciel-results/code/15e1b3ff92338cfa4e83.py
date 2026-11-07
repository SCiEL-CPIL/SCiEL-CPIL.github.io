"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: how many standard deviations the mean is below the best
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical overflow/underflow in cdf/pdf calculations
    # norm.cdf(-10) ~ 0, norm.pdf(10) ~ 0
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term proportional to uncertainty.
    # This helps the acquisition function to not collapse to zero in regions
    # where the GP is confident (low sigma) but might be slightly wrong, 
    # or to provide a gradient for exploration in early stages.
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration bonus
    total_improvement = ei + exploration_bonus
    
    # Cost-aware normalization
    # We want to maximize improvement per unit cost.
    # cost is given as (2000,), values > 1.
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = total_improvement / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
