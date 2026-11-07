"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, keep a safety margin of ~10s
_DEADLINE = time.time() + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters
    ----------
    mu : array, shape (2000,)
        GP mean predictions in objective units.
    sigma : array, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : array, shape (2000,)
        Evaluation cost for each candidate: 1 + mean(x_scaled_to_01).
    
    Returns
    -------
    scores : array, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid zero/negative, clip cost to avoid zero/negative
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Deadline guard: if we're close to the limit, return a simple, safe score
    if time.time() > _DEADLINE - 5.0:
        # Fallback: just use expected improvement per unit cost, but simple
        z = np.clip((f_best - mu) / sigma, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei / cost, 0.0)
    
    # Expected Improvement (EI)
    z = np.clip((f_best - mu) / sigma, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Also add a small exploration bonus to encourage exploring high-uncertainty regions
    # The uncertainty is captured by sigma, but we can add a small bonus
    # to avoid getting stuck in exploitation-only mode
    exploration_bonus = 0.01 * sigma / cost
    
    scores = ei / cost + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
