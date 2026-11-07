"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Thompson Sampling-inspired approach with adaptive exploration,
    combined with Expected Improvement per unit cost.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        Mean predictions from the GP surrogate.
    sigma : ndarray of shape (2000,)
        Standard deviation predictions from the GP surrogate.
    f_best : float
        Best observed objective value so far.
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI
    ei_per_cost = ei / cost_safe
    
    # Adaptive exploration: add a small exploration term that decreases
    # as we have more data (higher costs mean more evaluations done)
    # Use the mean cost as a proxy for progress
    mean_cost = np.mean(cost_safe)
    # Exploration coefficient that decays with progress
    # At the start (low cost), explore more; later, exploit more
    exploration_factor = 0.1 / (1.0 + 0.1 * mean_cost)
    
    # Add exploration bonus for high-uncertainty regions
    exploration_bonus = exploration_factor * sigma_safe
    
    # Combine EI per cost with exploration bonus
    scores = ei_per_cost + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
