import numpy as np
from scipy.stats import norm
import time

# Global state to track progress
_acq_call_count = 0
_acq_start_time = None

def acquisition(mu, sigma, f_best):
    global _acq_call_count, _acq_start_time
    
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Deadline guard: if we are close to the limit, return a safe score
    # The limit is 330s. We check if we've been running for too long.
    # However, this function is fast. The GP update is slow.
    # We just ensure we don't hang.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
        
    # Increment call count
    _acq_call_count += 1
    
    # Define a schedule for kappa
    # Assume ~25 acquisition calls (30 evals - 5 initial)
    # We want to explore early, exploit late.
    max_calls = 25
    current_progress = min(_acq_call_count / max_calls, 1.0)
    
    # Kappa starts high (exploration) and ends low (exploitation)
    # Typical UCB kappa is between 1 and 5.
    kappa_max = 4.0
    kappa_min = 0.5
    kappa = kappa_max - (kappa_max - kappa_min) * current_progress
    
    # UCB for minimization:
    # We want to minimize f.
    # GP posterior: f ~ N(mu, sigma^2)
    # We estimate the lower bound of f as mu - kappa * sigma.
    # We want to pick the point with the lowest lower bound.
    # So we maximize the score: - (mu - kappa * sigma) = -mu + kappa * sigma
    
    scores = -mu + kappa * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
