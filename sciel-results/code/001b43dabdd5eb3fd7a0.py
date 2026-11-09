import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_START_TIME = None

def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 295.0  # 5s safety margin before 300s
    return _DEADLINE

def _time_left():
    return max(0.0, _ensure_deadline() - time.time())

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an exploration bonus (Sigma) per unit cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Check deadline
    if _time_left() < 0.1:
        # If time is almost up, return a safe, finite array
        # This shouldn't happen for 2000 points, but just in case
        return np.zeros(len(mu), dtype=np.float64)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: Uncertainty (sigma)
    # In 8D, we need to balance exploitation (EI) with exploration (sigma)
    # to avoid local optima. The inspiration suggests sigma/cost was effective.
    # We combine them: (EI + beta * sigma) / cost
    beta = 0.5
    
    combined = ei + beta * sigma
    
    # Cost-aware: normalize by cost
    scores = combined / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
