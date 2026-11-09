import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu.copy()
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive beta for LCB
    # Start with high exploration, reduce as we find better points
    # f_best starts around 25 (max x^2) and goes to 0
    # Scale beta from 5.0 down to 1.0
    f_scale = np.clip(f_best / 25.0, 0.0, 1.0)
    beta = 1.0 + 4.0 * f_scale
    
    scores = -mu + beta * sigma_safe
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
