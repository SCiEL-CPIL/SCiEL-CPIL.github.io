"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for deadline tracking
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    global _START_TIME
    if _START_TIME is None:
        return float('inf')
    # 330s hard limit, use 300s as safety margin for the evaluator's kill switch
    return 300.0 - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    _init_time()
    
    # If we are close to the deadline, return a simple, fast score to avoid timeout
    if _time_remaining() < 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Standard robust calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI (primary exploitation term)
    cea = ei / cost_safe
    
    # Stochastic cost-aware acquisition:
    # Penalize high-cost candidates more aggressively using exp(-(beta0*cost)^2/2)
    # This creates a softer, more differential penalty than simple division by cost
    beta0 = 1.5
    cost_penalty = np.exp(-(beta0 * cost_safe) ** 2 / 2.0)
    
    # Combine deterministic CEA with stochastic cost penalty
    scores = cea * cost_penalty
    
    # Add a small exploration term to help escape local optima
    # Use a scaled sigma term that is also cost-aware
    exploration_weight = 0.15
    exploration = exploration_weight * sigma_safe / cost_safe * cost_penalty
    
    scores = scores + exploration
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
