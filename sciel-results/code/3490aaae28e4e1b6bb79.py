# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using cost-adjusted Lower Confidence Bound (LCB).
    
    Strategy:
    - Use LCB formulation: score = -mu + beta * sigma / cost
    - The exploration term (sigma) is scaled by 1/cost, penalizing high-cost
      regions more heavily in the uncertainty bonus.
    - The mean term (-mu) remains invariant to cost, ensuring exploitation
      signal is preserved.
    - This creates a non-uniform exploration landscape that prioritizes
      high-uncertainty, low-cost areas early in the budget.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    scores: (2000,) array of finite scores (higher = evaluate next)
    """
    elapsed = _get_elapsed()
    
    # Safety margin: if we're getting close to the 330s limit, return a simple score
    if elapsed > 320:
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_safe = np.asarray(mu, dtype=np.float64)
        score = -mu_safe + 3.0 * sigma_safe / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure arrays are floats and handle potential NaNs or Infs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-adjusted LCB:
    # We want to minimize mu - beta * sigma / cost
    # So we maximize: -mu + beta * sigma / cost
    #
    # The exploration term (uncertainty bonus) is scaled by 1/cost,
    # which penalizes high-cost regions more heavily in the uncertainty bonus.
    # The mean term remains invariant.
    beta = 3.0
    
    # Calculate the cost-adjusted LCB score
    score = -mu + beta * sigma_safe / cost_safe
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
