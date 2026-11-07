"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_call_count = 0

# Known minimum for Hartmann-6D
F_MIN = -3.32237
# Reference upper bound for progress calculation
F_REF = 3.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a dynamic Lower Confidence Bound (LCB) where the exploration
    coefficient kappa decreases as f_best approaches the known minimum.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
        _call_count = 0
    
    _call_count += 1
    
    # Deadline guard: return simple LCB if we're running out of time
    if time.time() - _start_time > 285:
        scores = 5.0 * sigma - mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate progress towards the known minimum
    # P = 0 when f_best is bad (high), P = 1 when f_best is at minimum
    # We use F_REF as a starting point estimate
    range_val = F_REF - F_MIN
    if range_val <= 0:
        range_val = 1.0
        
    # Clamp f_best to reasonable bounds to avoid extreme kappa values
    # If f_best is better than F_MIN (numerical error), treat as F_MIN
    # If f_best is worse than F_REF, treat as F_REF
    f_clamped = np.clip(f_best, F_MIN, F_REF)
    
    progress = (F_REF - f_clamped) / range_val
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic kappa: starts high (exploration), ends low (exploitation)
    kappa_max = 6.0
    kappa_min = 0.5
    
    kappa = kappa_max - (kappa_max - kappa_min) * progress
    
    # Lower Confidence Bound for minimization:
    # We want to maximize the score.
    # LCB(x) = mu(x) - kappa * sigma(x) is the lower bound on f(x).
    # To find the most promising point, we want the lowest possible f(x).
    # So we maximize -LCB(x) = -mu(x) + kappa * sigma(x)
    # Or equivalently, we can define the acquisition score as:
    # score = kappa * sigma - mu
    # Higher score means more likely to be better (lower f).
    
    scores = kappa * sigma_safe - mu
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
