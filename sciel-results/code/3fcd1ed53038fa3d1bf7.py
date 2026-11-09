import numpy as np
import time

# ---- Deadline guard ----
_START_TIME = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 20.0

# ---- Constants ----
DIM = 4
RANGE_LOW = -4.0
RANGE_HIGH = 5.0
SPAN = RANGE_HIGH - RANGE_LOW
COST_BUDGET = 30.0
N_INIT = 5
N_CAND = 2000

def _compute_scores(mu, sigma, cost, f_best):
    """
    Computes cost-aware acquisition scores.
    Uses a linear cost-normalized regret lower bound:
    score = (f_best - mu + sigma * 0.5) / cost
    This balances exploitation (f_best - mu) and exploration (sigma)
    while accounting for the cost of evaluation.
    """
    eps = 1e-8
    
    # Ensure sigma is positive and finite
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    
    # Ensure cost is positive and finite
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute score
    # The term (f_best - mu) drives exploitation towards better points.
    # The term (sigma * 0.5) drives exploration towards uncertain regions.
    # Dividing by cost prioritizes low-cost evaluations.
    # We clip the numerator to be non-negative to avoid negative scores
    # which would imply "worse than nothing" but might be numerically unstable
    # or confusing if the optimizer expects higher is better for all valid candidates.
    # Actually, if mu > f_best, the improvement is negative.
    # However, in BO, we usually only care about points that *might* improve.
    # But the formula `f_best - mu + 0.5*sigma` can be negative if mu is much larger than f_best.
    # Should we clip it?
    # If we clip to 0, we ignore points that are predicted to be worse.
    # This is standard for EI/UCB-like functions to focus on potential improvements.
    # Let's clip the numerator to 0 to ensure we only select candidates that
    # have a reasonable chance of being better than the current best,
    # weighted by uncertainty.
    
    numerator = f_best - mu_safe + 0.5 * safe_sigma
    numerator = np.maximum(numerator, 0.0)
    
    score = numerator / safe_cost
    
    # Handle any remaining numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Public entry point.
    """
    # Hard deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: just prefer low cost
        cost_arr = np.asarray(cost, dtype=np.float64)
        return 1.0 / np.where(cost_arr < 1e-9, 1e-9, cost_arr)
    
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    return _compute_scores(mu_arr, sigma_arr, cost_arr, float(f_best))
