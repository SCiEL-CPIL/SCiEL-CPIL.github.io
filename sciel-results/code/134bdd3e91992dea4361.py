"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()
    MAX_TIME = 290.0  # Safety margin before 300s kill

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)

    # Expected Improvement (EI)
    ei = diff * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)

    # Exploration term: Reward high uncertainty
    # This helps escape local optima or unexplored regions where EI might be low
    # because mu > f_best.
    explore_term = sigma_safe
    
    # Blend EI and Exploration
    # alpha controls the weight of exploration
    # A value of 0.5-1.0 is typical for small budgets
    alpha = 0.5 
    combined_score = ei + alpha * explore_term

    # Cost-aware normalization
    score = combined_score / cost_safe

    # Check time deadline
    elapsed = time.time() - t_start
    if elapsed > MAX_TIME:
        # Return best result found so far (which is the current score array)
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
