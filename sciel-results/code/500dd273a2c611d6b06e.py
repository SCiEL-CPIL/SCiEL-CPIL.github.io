"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for time management
_start_time = None


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _time_remaining():
    """Return remaining time in seconds, with safety margin."""
    elapsed = time.time() - _get_start_time()
    # Safety margin: stop at 320s (10s buffer before 330s kill)
    return max(0.0, 320.0 - elapsed)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a cost-normalized Lower Confidence Bound (LCB) acquisition:
    score = (mu - kappa * sigma) / cost
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check time budget - if very little time left, return simple fallback
    remaining = _time_remaining()
    if remaining < 2.0:
        # Simple LCB fallback
        kappa = 1.5
        scores = (mu - kappa * sigma_safe) / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute LCB: for minimization, lower mu and higher sigma are better
    # We want to maximize (mu - kappa*sigma) / cost
    # Higher kappa means more exploration
    
    # Adaptive kappa: start with more exploration early, reduce as we converge
    # Use f_best to gauge progress - if f_best is close to known minimum (-78.33), reduce exploration
    # But we don't know exact minimum, so use a heuristic based on cost spent
    # Cost spent so far is implicit in the loop; use a fixed reasonable kappa
    
    # For Styblinski-Tang 2D, the function has a single global minimum
    # A moderate kappa works well. Let's use 1.5 which balances exploitation/exploration
    kappa = 1.5
    
    # Compute LCB scores
    lcb = mu - kappa * sigma_safe
    scores = lcb / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
