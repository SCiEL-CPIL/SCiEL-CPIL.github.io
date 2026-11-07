"""Rewritten program using Cost-Weighted UCB for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted UCB.
    
    Parameters:
    mu: array-like, mean predictions from GP (in objective units)
    sigma: array-like, standard deviation predictions from GP (in objective units)
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # UCB for minimization:
    # We want to maximize the expected improvement plus an exploration bonus.
    # Expected improvement in mean: f_best - mu
    # Exploration bonus: beta * sigma
    # Total optimistic value: (f_best - mu) + beta * sigma
    
    # Beta parameter controls exploration. 
    # For small budgets (total cost < 30), high exploration is needed.
    # Beta = 2.0 is a reasonable default for UCB in BO.
    beta = 2.0
    
    exploitation = f_best - mu
    exploration = beta * sigma_safe
    
    # Optimistic value
    optimistic_value = exploitation + exploration
    
    # Cost-aware normalization
    # We divide by cost to prefer cheaper evaluations that offer similar optimistic value.
    scores = optimistic_value / cost_safe
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
