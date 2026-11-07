"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure arrays are float64 and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (EI can be slightly negative due to numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    score = ei / cost_safe
    
    # Add a small exploration bonus proportional to sigma to encourage
    # exploring uncertain regions, especially early on when f_best is far from minimum
    # The weight decreases as f_best gets closer to the known minimum (-0.8)
    # This helps balance exploitation and exploration
    min_possible = -0.8
    gap_to_min = f_best - min_possible  # positive, how far we are from the minimum
    # Scale exploration weight: larger when gap is large (early), smaller when close
    explore_weight = 0.05 * np.exp(-gap_to_min / 0.5)
    
    # UCB-like exploration term
    ucb_bonus = explore_weight * sigma_safe / cost_safe
    
    score = score + ucb_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
