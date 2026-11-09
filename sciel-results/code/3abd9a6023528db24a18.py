"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# Each call to acquisition should be fast, but we keep a guard just in case.
_DEADLINE = 320.0  # safety margin
_start_time = time.time()


def _check_time():
    """Return True if we should abort (time exceeded)."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best observed objective value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    if _check_time():
        # Fallback: return uniform scores to avoid invalid output
        return np.ones(2000)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Make EI non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware acquisition with gamma parameter
    # Using a power of cost to balance exploration vs exploitation
    # Lower gamma = more weight on high-improvement regions (exploitation)
    # Higher gamma = more weight on cheap regions (exploration)
    gamma = 1.0
    scores = ei / (cost_safe ** gamma)
    
    # Handle any NaN or inf values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
