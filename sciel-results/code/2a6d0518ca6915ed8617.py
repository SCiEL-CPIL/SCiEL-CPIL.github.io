import numpy as np
from scipy.stats import norm
import time

# State for tracking across calls within a single seed run
_state = {
    'n': 0,
    'prev_f_best': None,
    'start_time': None,
    'seed_id': 0
}

N_TOTAL = 25.0
TIME_LIMIT = 280.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-9)
    
    # Initialize time tracking
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    elapsed = time.time() - _state['start_time']
    
    # Detect new seed: f_best increased significantly (reset) or first call
    prev = _state['prev_f_best']
    if prev is None or f_best > prev + 1e-6:
        _state['n'] = 0
        _state['start_time'] = time.time()  # Reset time for new seed
        elapsed = 0.0
    
    _state['prev_f_best'] = f_best
    n = _state['n']
    _state['n'] = n + 1
    
    # Progress: 0 to 1 over the 25 optimization steps (excluding initial 5)
    # n goes from 0 to 24 for the 25 optimization calls
    p = min(n / N_TOTAL, 1.0)
    
    # Time-aware exploration: reduce exploration as time runs out
    time_factor = max(0.0, 1.0 - elapsed / TIME_LIMIT)
    
    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB-style exploration term
    # Use a beta that decays with progress and time pressure
    # Early: high exploration, Late: low exploration
    beta = 2.0 * (1.0 - p) * (1.0 - p) * time_factor
    
    # Also add a small constant exploration to avoid getting stuck
    base_exploration = 0.1 * (1.0 - p)
    
    score = ei + (beta + base_exploration) * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
