"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Expected Improvement per unit cost with exploration penalty.
    
    mu, sigma: arrays of shape (2000,) - GP mean and std in objective units
    f_best: float - lowest observed value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative costs
    cost = np.maximum(cost, 1e-6)
    
    # Standard deviation guard
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    score = ei / cost
    
    # Exploration term: reward high uncertainty in cheap regions
    # This helps escape local optima by favoring unexplored cheap areas
    # The exploration weight decays with cost to avoid expensive exploration
    exploration_weight = 0.5
    exploration = exploration_weight * sigma / cost
    
    # Combine exploitation and exploration
    total_score = score + exploration
    
    # Safety: ensure all values are finite
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    return total_score
# EVOLVE-BLOCK-END
