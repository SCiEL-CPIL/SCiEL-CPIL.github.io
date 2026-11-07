import numpy as np
import time

_step_count = 0
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _step_count, _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return scores that favor exploitation (low mu) to finish quickly?
        # Or just 0.
        return np.zeros(len(mu))

    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta based on f_best
    # f_best is the current minimum observed value.
    # Ackley min is 0. Max is ~30.
    # If f_best is high, we are early -> high beta (explore).
    # If f_best is low, we are late -> low beta (exploit).
    
    # Normalize f_best to [0, 1]
    # Assume range 0 to 20 is significant.
    progress = np.clip(f_best / 20.0, 0.0, 1.0)
    
    # Beta ranges from 0.5 (exploit) to 3.0 (explore)
    beta = 0.5 + 2.5 * progress
    
    # UCB for minimization: maximize (-mu + beta * sigma)
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    _step_count += 1
    return scores
