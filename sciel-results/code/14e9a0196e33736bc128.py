"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that balances expected improvement
    against evaluation cost, with exploration bonuses for high-uncertainty
    regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed objective value so far
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Ensure inputs are finite and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle cases where EI might be negative or very small
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get EI per unit cost
    ei_per_cost = ei / cost
    
    # Add exploration bonus: reward high uncertainty regions
    # This helps escape local optima and explore the space more broadly
    # The bonus is scaled to be comparable to EI_per_cost
    max_ei_per_cost = np.max(ei_per_cost) if len(ei_per_cost) > 0 else 1.0
    if max_ei_per_cost > 1e-12:
        exploration_weight = 0.3  # Weight for exploration
        # Exploration bonus based on sigma, normalized
        sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
        if sigma_max > 1e-12:
            exploration_bonus = exploration_weight * (sigma / sigma_max) * max_ei_per_cost
        else:
            exploration_bonus = np.zeros_like(sigma)
    else:
        exploration_bonus = np.zeros_like(sigma)
    
    # Combine exploitation (EI per cost) and exploration
    scores = ei_per_cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
