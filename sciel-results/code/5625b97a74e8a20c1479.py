"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: The evaluator kills the program after 330s wall-clock.
# We set a safety margin to return a result before the hard kill.
_DEADLINE = 325.0
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
    # Time guard: If we are running out of time, return a safe default.
    # Returning zeros or uniform scores is better than crashing or hanging.
    if _check_time():
        return np.zeros(2000, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs to prevent division by zero or invalid operations
    # Sigma must be positive for the EI formula
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost must be positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Using scipy.stats.norm for accuracy
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI should theoretically be non-negative, but numerical errors can make it slightly negative.
    # We clip to 0 to ensure we don't penalize points that are effectively at the best value.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost_safe
    
    # Handle any remaining non-finite values (NaN, Inf) by replacing them with 0.
    # This ensures the output is always finite and valid.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
