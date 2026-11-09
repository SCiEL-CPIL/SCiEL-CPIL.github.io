"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) per unit cost - standard exploitation
    2. Uncertainty-weighted term - balances exploration
    3. Safety margin against numerical issues
    
    The acquisition function is designed to be robust and handle edge cases
    where sigma is very small or cost varies significantly.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: where sigma is very small, use a simpler heuristic
    eps = 1e-12
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, eps)
    ei_per_cost = ei / cost_safe
    
    # Add a small uncertainty bonus to encourage exploration in uncertain regions
    # This helps avoid getting stuck in local optima
    uncertainty_bonus = sigma_safe / cost_safe * 0.1
    
    # Combine: primarily EI per cost, with a small exploration bonus
    score = ei_per_cost + uncertainty_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative scores
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
