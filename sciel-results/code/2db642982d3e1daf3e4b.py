"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: kill after 330s wall-clock, use 300s as safety margin
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds


def _check_deadline():
    """Return True if we should stop working (deadline approached)."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline
    if _check_deadline():
        # Return a simple heuristic to stay within time budget
        return -(mu / (cost + 1e-9))
    
    # Compute z-score for expected improvement
    improvement = f_best - mu
    z = improvement / sigma
    
    # Expected Improvement
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: use cost^0.7 instead of cost^1 to be less extreme
    cost_factor = np.power(cost, 0.7)
    
    # Base acquisition: EI per unit cost (with softened cost)
    score = ei / cost_factor
    
    # Add a small exploration bonus for high uncertainty
    # This helps escape local optima in the rosenbrock landscape
    exploration_weight = 0.1
    score += exploration_weight * sigma / cost_factor
    
    # Gentle high-cost penalty to nudge toward cost-efficiency
    # Points with very high cost get a small additional penalty
    cost_penalty = 0.05 * np.maximum(cost - 2.0, 0.0) / cost_factor
    score -= cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
