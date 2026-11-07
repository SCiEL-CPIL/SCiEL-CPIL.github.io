"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean predictions of the Gaussian Process
    sigma : array-like, shape (n,)
        Standard deviation predictions of the Gaussian Process
    f_best : float
        Best observed function value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores for each candidate point
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're getting close to the 300s limit,
    # fall back to a simple deterministic score to ensure we return in time
    elapsed = time.time() - _start_time
    if elapsed > 290.0:
        # Fallback: return cost-normalized sigma (pure exploration)
        cost_safe = np.maximum(cost, 1e-9)
        scores = sigma / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Ensure PI is in [0, 1]
    pi = np.clip(pi, 0.0, 1.0)
    
    # Cost-aware score: PI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = pi / cost_safe
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps avoid premature convergence
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    scores = scores + exploration_bonus
    
    # Ensure all values are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
