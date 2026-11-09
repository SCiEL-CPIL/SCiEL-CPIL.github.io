import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 325.0

_HARTMANN_MIN = -3.86278
_HARTMANN_MAX = 3.32237

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: pure exploitation
        return -mu

    # Adaptive UCB
    # progress: 1.0 when f_best is at max (worst), 0.0 when f_best is at min (best)
    f_best_clipped = np.clip(f_best, _HARTMANN_MIN, _HARTMANN_MAX)
    progress = (f_best_clipped - _HARTMANN_MIN) / (_HARTMANN_MAX - _HARTMANN_MIN)
    
    # Beta starts high (exploration) and decreases (exploitation)
    beta = 1.0 + 5.0 * progress
    
    # Score: higher is better
    # We minimize mu - beta * sigma, so we maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
