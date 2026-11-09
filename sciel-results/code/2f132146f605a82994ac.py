"""Cost-aware acquisition for Bayesian optimization on Styblinski-Tang 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s margin before 300s limit


def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions
    sigma : array_like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite inputs
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-12)
    
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: just use negative mu (exploit best)
        return -mu
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring: use a Lagrangian-style approach
    # score = EI / cost is the classic approach, but let's try a hybrid
    # that also considers the raw improvement potential
    
    # Primary score: EI per unit cost
    score = ei / cost
    
    # Add a small exploration bonus for high uncertainty regions
    # weighted by inverse cost to avoid penalizing exploration too heavily
    exploration_bonus = sigma / cost * 0.1
    
    # Combine
    scores = score + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
