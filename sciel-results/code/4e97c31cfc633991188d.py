import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT - _SAFETY_MARGIN:
        # Return a simple, fast fallback score
        # Just return -mu + sigma to be safe and fast
        return -mu + sigma

    # Adaptive Kappa
    # f_best is the minimum observed value.
    # Weierstrass 1D on [-0.5, 0.5] has min 0.
    # Typical values are in [0, ~1.5].
    # If f_best is high, we need more exploration.
    # If f_best is low, we need more exploitation.
    
    # Normalize f_best roughly. 
    # If f_best > 1.0, assume early stage.
    # If f_best < 0.1, assume late stage.
    # Let's use a smooth decay.
    
    # A simple heuristic:
    # beta = 2.0 + 2.0 * max(0, 1.0 - f_best)
    # If f_best = 1.5, beta = 2.0
    # If f_best = 0.5, beta = 3.0
    # If f_best = 0.0, beta = 4.0 -> This might be too high?
    
    # Let's try a fixed beta first, as adaptive without step count can be noisy.
    # The inspiration said beta=3.8 improved a score.
    # Let's use beta = 2.5.
    
    beta = 2.5
    
    # Score = -mu + beta * sigma
    # Ensure sigma is non-negative (it is std dev, so it should be)
    # Clip sigma to avoid numerical issues if any negative values appear due to precision
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
