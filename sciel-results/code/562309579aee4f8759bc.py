"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we don't exceed the deadline
_start_time = time.time()
_DEADLINE = 300.0  # 300s limit, keep safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and an exploration bonus
    that accounts for the cost structure. The key insight is that points with
    lower cost should be preferred when the expected improvement is similar,
    but we also want to explore uncertain regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a simple fallback to avoid timeout
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring:
    # We want to balance exploration (high sigma) and exploitation (low mu)
    # while accounting for the cost of evaluation.
    
    # Primary: EI per unit cost
    ei_per_cost = ei / cost
    
    # Secondary: Add a small exploration bonus for high uncertainty regions
    # This helps escape local minima in the acquisition landscape
    exploration_bonus = 0.1 * sigma / cost
    
    # Combine: weighted sum
    # The weight on EI_per_cost dominates, but exploration_bonus provides
    # a tie-breaker that favors cheaper, more uncertain points
    score = ei_per_cost + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
