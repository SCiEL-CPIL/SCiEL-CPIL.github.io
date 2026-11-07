import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        return -mu

    # Adaptive Beta:
    # Start with high exploration (beta=4.0) to escape local minima early.
    # Decay to lower exploitation (beta=1.0) as we converge.
    # The decay is exponential with a time constant of 5 evaluations.
    beta = 1.0 + 3.0 * np.exp(-_eval_count / 5.0)
    
    scores = -mu + beta * sigma
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
