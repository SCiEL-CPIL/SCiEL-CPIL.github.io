"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time to enforce deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 30.0  # seconds of safety margin before 330s limit


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing levy 2d.
    
    Uses a hybrid approach:
    1. Cost-adjusted Expected Improvement (EI) as the primary signal
    2. A small exploration bonus for high-uncertainty regions
    3. Deadline guard to ensure we return before timeout
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of GP posterior at candidate points
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Check deadline - if too close to limit, return simple UCB-based scores quickly
    if _elapsed() > 300.0:
        # Fallback: simple cost-aware LCB
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        beta = 2.0
        scores = -(mu - beta * sigma_safe) / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure safe values
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Use stable computation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Cost-adjusted EI: improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration term to encourage visiting uncertain regions
    # This helps escape local optima and explore the space
    # Use a smooth saturation to avoid over-exploration
    exploration_weight = 0.005
    exploration = exploration_weight * sigma_safe / (sigma_safe + 0.1)
    
    # Combine exploitation and exploration
    scores = ei_per_cost + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
