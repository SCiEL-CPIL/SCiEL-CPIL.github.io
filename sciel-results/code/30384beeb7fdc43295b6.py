import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score (e.g., just -mu)
        # to force a selection quickly.
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive Beta Strategy
    # Total acquisition calls expected: 25 (30 total evals - 5 initial)
    total_calls = 25.0
    # Clamp progress between 0 and 1
    progress = np.clip(_eval_count / total_calls, 0.0, 1.0)
    
    # Start with high exploration (beta=2.5) and end with exploitation (beta=0.5)
    beta_start = 2.5
    beta_end = 0.5
    
    # Linear interpolation of beta
    beta = beta_start * (1.0 - progress) + beta_end * progress
    
    # LCB for minimization: -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
