"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: return early if we're close to the 330s limit
_DEADLINE_START = None
_DEADLINE_LIMIT = 315.0  # 15s safety margin

def _check_deadline():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return (time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive LCB (Lower Confidence Bound).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (minimum)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1])
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next)
    """
    # Deadline guard
    if _check_deadline():
        # Return a reasonable fallback: prioritize low cost and low mu
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive LCB approach:
    # score = -mu - beta * sigma, but we want higher = better
    # So we use: score = (-mu - beta * sigma) / cost
    # where beta adapts to the current uncertainty level
    
    # Compute adaptive beta based on the spread of sigma
    # When sigma is large (early on), we explore more; when small, we exploit
    max_sigma = np.max(sigma)
    if max_sigma < 1e-9:
        max_sigma = 1e-9
    
    # Beta scales with the uncertainty: higher uncertainty -> more exploration
    # Use a logarithmic scaling to avoid extreme values
    # beta ranges from ~0.5 (low uncertainty) to ~3.0 (high uncertainty)
    # We normalize sigma by max_sigma to get a relative uncertainty measure
    rel_uncertainty = sigma / max_sigma
    
    # Adaptive beta: base + scaled component
    # When rel_uncertainty is high, beta is larger (more exploration)
    beta = 0.5 + 2.5 * np.clip(rel_uncertainty, 0, 1)
    
    # LCB score: we want to minimize mu + beta * sigma, so maximize -(mu + beta * sigma)
    # Divide by cost to account for evaluation cost
    lcb_value = mu + beta * sigma
    score = -lcb_value / cost
    
    # Add a small bonus for points with very low cost to encourage cheap evaluations
    # This helps in the early stages when we haven't explored much
    cost_bonus = 0.1 / cost
    
    score = score + cost_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
