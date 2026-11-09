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
        
        ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        
        # Cost-aware: divide by cost^1.5
        score = _safe_divide(ei, np.power(cost, 1.5), eps=1e-9)
        
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score
