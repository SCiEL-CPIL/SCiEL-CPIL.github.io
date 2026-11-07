import numpy as np
from scipy.stats import norm
import time

_call_count = 0
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        _call_count += 1
        return -mu  # Simple fallback
    
    _call_count += 1
    
    # Adaptive LCB: beta decays over time
    # 5 initial + 25 optimization = 30 total evaluations
    # call_count goes from 1 to 25 for optimization steps
    t = _call_count
    # Start with high exploration, decay to exploitation
    beta = max(0.5, 2.0 * np.exp(-0.1 * t))
    
    # LCB for minimization: lower score means more promising
    # We want higher return value = more promising
    # So: score = -(mu - beta * sigma) = -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
