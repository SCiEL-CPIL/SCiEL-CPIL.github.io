import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _safe_divide(a, b, eps=1e-9):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    if time.time() - _start_time > 295:
        # Fallback to a simple, fast score
        # Just return EI / cost to be safe
        sigma_safe = np.maximum(sigma, 1e-9)
        diff = f_best - mu
        z = _safe_divide(diff, sigma_safe)
        z = np.clip(z, -10, 10)
        ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        return _safe_divide(ei, cost_safe)

    sigma = np.maximum(sigma, 1e-9)
    diff = f_best - mu
    z = _safe_divide(diff, sigma)
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Uncertainty penalty
    # Higher sigma -> lower score.
    # This encourages the algorithm to pick points where the GP is more confident
    # that there is an improvement, reducing wasted evaluations in high-noise areas.
    alpha = 1.0
    uncertainty_penalty = np.exp(-alpha * sigma)
    
    # Cost factor
    # Using cost^0.75 instead of sqrt(cost) or cost.
    cost_factor = cost_safe ** 0.75
    
    score = _safe_divide(ei, cost_factor) * uncertainty_penalty
    
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
