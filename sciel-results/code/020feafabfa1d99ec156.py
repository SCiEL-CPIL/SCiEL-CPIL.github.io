import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s


def _check_deadline():
    """Return True if we should abort and return a fallback immediately."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) for minimization,
    with an adaptive LCB blend that emphasizes exploration early
    (when f_best is high) and exploitation later.
    
    mu, sigma : (2000,) arrays, sigma is the GP posterior std dev
    f_best : lowest observed value so far
    Returns: (2000,) scores, higher = more promising
    """
    # Handle edge cases
    if sigma is None or len(mu) != len(sigma):
        return -mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Check deadline
    if _check_deadline():
        return -mu + sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = z * norm.pdf(z) + norm.cdf(z)
    ei = ei * sigma
    
    # Adaptive LCB: use f_best as a stage indicator
    # Branin range: f_best starts around 30-50, ends near 0.4
    # stage = 1.0 early (high f_best), 0.0 late (low f_best)
    f_max_est = 50.0
    stage = np.clip(f_best / f_max_est, 0.0, 1.0)
    
    # Kappa for LCB: high early (explore), low late (exploit)
    kappa = 0.5 + 1.5 * stage
    
    # LCB score: -mu + kappa * sigma
    lcb = -mu + kappa * sigma
    
    # Blend: use EI more as we progress (stage decreases)
    # w_ei goes from ~0.3 early to ~1.0 late
    w_ei = 1.0 - 0.7 * stage
    w_lcb = 1.0 - w_ei
    
    # Combine scores
    scores = w_ei * ei + w_lcb * lcb
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
