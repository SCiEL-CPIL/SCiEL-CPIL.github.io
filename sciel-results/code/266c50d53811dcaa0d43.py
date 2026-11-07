"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Standard Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add an exploration bonus proportional to the standard deviation.
    # This encourages the algorithm to explore regions with high uncertainty,
    # which is crucial in the early stages of Bayesian Optimization with a small budget.
    # The weight 0.5 balances exploitation (EI) and exploration (sigma).
    exploration_weight = 0.5
    numerator = ei + exploration_weight * sigma_safe
    
    # Cost-aware scoring: higher score is better
    # Cost is >= 1.0, so division is safe.
    score = numerator / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
