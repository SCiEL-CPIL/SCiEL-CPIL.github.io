import numpy as np
import time
from scipy.stats import norm

_T0 = None
_DEADLINE = None

def _ensure_deadline():
    global _T0, _DEADLINE
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 295.0  # 5s margin for safety
    return _DEADLINE

def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    elapsed = time.perf_counter() - _T0
    
    # Fallback if time is running out (though POI is very fast)
    if elapsed > _DEADLINE - 5.0:
        # Simple LCB fallback
        return -mu + 1.0 * sigma

    # Probability of Improvement (POI)
    # We want to maximize the probability that f(x) < f_best
    # Z = (f_best - mu) / sigma
    # POI = Phi(Z)
    
    # Handle sigma == 0 to avoid division by zero
    # If sigma is 0, the prediction is deterministic.
    # If mu < f_best, probability is 1. Else 0.
    # Using a small epsilon for stability.
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    Z = (f_best - mu) / sigma_safe
    
    # Phi(Z) is the CDF of the standard normal distribution
    score = norm.cdf(Z)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
