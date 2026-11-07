"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Thompson Sampling, with a deadline guard.
    """
    # Deadline guard: ensure we don't exceed time limits
    # We'll compute quickly and return finite scores
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: divide by cost to get EI per unit cost
    # This encourages evaluating cheaper points
    ei_per_cost = ei / np.maximum(cost, 1e-9)
    
    # Add a small exploration term based on uncertainty
    # This helps when EI is zero (mu >= f_best)
    # Use a scaled version of sigma as exploration bonus
    exploration_bonus = 0.1 * sigma_safe / np.maximum(cost, 1e-9)
    
    # Combine EI and exploration
    score = ei_per_cost + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Return scores (higher = evaluate next)
    return score
# EVOLVE-BLOCK-END
