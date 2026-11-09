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
    if time.time() - _start_time > 320:
        return np.zeros_like(mu, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    diff = f_best - mu
    z = _safe_divide(diff, sigma)
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Blend EI and PI for robustness in sparse, multimodal landscapes
    # PI is more robust for exploration; EI is sensitive to high uncertainty
    # A weighted combination balances exploitation (EI) and exploration (PI)
    alpha = 0.5
    blended = alpha * ei + (1 - alpha) * pi * (diff + sigma)
    
    # Cost-aware: divide by cost^1.5 (as in current program)
    # This reduces bias towards low-cost regions while still being cost-aware
    score = _safe_divide(blended, np.power(cost, 1.5), eps=1e-9)
    
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
