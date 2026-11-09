"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_t0 = time.time()
_DEADLINE = 325.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Returns 2000 finite scores (higher = evaluate next).
    
    Strategy: Cost-aware Expected Improvement with log-cost penalty
    and a small exploration bonus for high-uncertainty points.
    """
    # Deadline guard
    if time.time() - _t0 > _DEADLINE:
        # Return uniform random scores to avoid invalidity
        return np.random.rand(2000)

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: use log(cost) to give diminishing penalty
    # This avoids over-penalizing expensive points while still preferring cheap ones
    cost_safe = np.maximum(cost, 1e-10)
    log_cost = np.log(cost_safe + 1.0)  # +1 to avoid log(0) and shift
    
    # Base score: EI divided by log cost
    score = ei / log_cost
    
    # Add a small exploration bonus for points with high uncertainty
    # This encourages exploring regions where the GP is uncertain
    exploration_bonus = 0.05 * sigma_safe / np.max(sigma_safe)
    score += exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
