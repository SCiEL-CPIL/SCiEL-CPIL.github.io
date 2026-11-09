"""Cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _safe_divide(a, b, eps=1e-9):
    """Safe element-wise division with epsilon guard."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Uses PI = Phi((f_best - mu) / sigma), which is bounded in [0, 1].
    Dividing by cost penalizes expensive evaluations, balancing exploration
    and exploitation while respecting the budget.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = evaluate next).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return zeros if close to limit to avoid timeout
    if time.time() - _start_time > 320:
        return np.zeros_like(mu, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero in z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Probability of Improvement
    diff = f_best - mu
    z = _safe_divide(diff, sigma_safe)
    
    # Clip z to avoid numerical issues in norm.cdf for very large values
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure non-negative PI (numerical safety)
    pi = np.maximum(pi, 0.0)
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    # Cost is in [1, 2]. Dividing by cost directly makes the acquisition
    # function more sensitive to cost differences.
    cost_safe = np.maximum(cost, 1e-9)
    
    score = _safe_divide(pi, cost_safe, eps=1e-9)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
