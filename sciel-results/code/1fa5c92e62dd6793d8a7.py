import numpy as np
from scipy.stats import norm
import time

_call_count = 0
_last_f_best = None
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _call_count, _last_f_best, _start_time
    
    # Safety margin check (though unlikely to trigger)
    if time.time() - _start_time > 320:
        # Return something valid, e.g., EI
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Reset logic for new seed/run
    if _last_f_best is not None and f_best > _last_f_best:
        _call_count = 0
    _last_f_best = f_best
    _call_count += 1
    
    # 1. Compute EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute UCB (for minimization: -mu + beta * sigma)
    beta = 2.5
    ucb = -mu + beta * sigma
    
    # 3. Normalize to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-9:
        ucb_norm = np.ones_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
        
    # 4. Adaptive Weight
    # _call_count is 1-based. 
    # We expect ~25 calls.
    # Transition from UCB (w=0) to EI (w=1)
    x = _call_count - 1 
    mid = 14
    k = 0.4
    w = 1.0 / (1.0 + np.exp(-k * (x - mid)))
    
    # 5. Combine
    score = w * ei_norm + (1.0 - w) * ucb_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
