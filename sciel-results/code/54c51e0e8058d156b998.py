import numpy as np
import time
from scipy.stats import norm

# Global state for adaptive acquisition
_start_time = time.time()
_DEADLINE = 325.0  # 300s limit, leave 5s margin
_eval_count = 0
_total_evals = 30  # From task description: 5 initial + 25 steps (30 total)

def acquisition(mu, sigma, f_best):
    global _eval_count
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback
        return -mu.copy()
    
    # Increment evaluation count
    _eval_count += 1
    
    # Adaptive Beta for UCB
    # Early on: high exploration (beta ~ 3-4)
    # Late on: high exploitation (beta ~ 0.5-1)
    # We have 30 total evaluations.
    # Let's use a smooth decay.
    progress = _eval_count / _total_evals
    # Beta starts at 4.0 and decays to 0.5
    beta = 4.0 * (1 - progress) + 0.5 * progress
    
    # UCB for minimization: score = -mu + beta * sigma
    # Higher score is better.
    scores = -mu + beta * sigma
    
    return scores
