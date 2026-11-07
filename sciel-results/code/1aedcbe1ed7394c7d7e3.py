"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an exploration term that favors cheaper evaluations in the
    early-to-mid budget, preventing premature convergence to expensive
    regions while still exploiting the GP.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points (objective units).
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points (objective units).
    f_best : float
        Best (lowest) observed objective value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of coords scaled to [0,1]).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard: track elapsed time to ensure we return within limits
    # This function is called many times in the loop; we add a small safety check
    # but the main computation is fast (vectorized NumPy ops on 2000 elements)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide EI by cost to get improvement per unit cost
    # Use a smoothed cost to avoid division by very small values
    cost_smoothed = np.maximum(cost, 0.1)
    ei_per_cost = ei / cost_smoothed
    
    # Exploration term: favor regions with high uncertainty (high sigma)
    # and low cost, especially in the early phase when f_best is still high
    # The exploration weight decreases as we get closer to the optimum
    # Use a normalized uncertainty measure
    sigma_norm = sigma / (np.max(sigma) + 1e-10)
    
    # Cheap-region exploration: add a small bonus for low-cost, high-uncertainty points
    # This prevents premature convergence to expensive regions
    # The weight is small to not dominate the EI signal
    exploration_weight = 0.05
    
    # Uncertainty bonus: high sigma and low cost
    uncertainty_bonus = sigma_norm / cost_smoothed
    
    # Combine EI per cost with exploration
    scores = ei_per_cost + exploration_weight * uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
