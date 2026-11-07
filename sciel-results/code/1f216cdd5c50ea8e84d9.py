"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Safety margin: The limit is 330s. We aim to finish within 320s to be safe.
_DEADLINE = _START_TIME + 320.0
_call_count = 0
_prev_f_best = None
_stall_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count, _prev_f_best, _stall_count
    
    # Explicit deadline guard
    if time.time() > _DEADLINE:
        # Fallback: Greedy selection based on mean
        return -mu
        
    _call_count += 1
    
    # Track progress to detect stalls
    if _prev_f_best is not None:
        if f_best >= _prev_f_best - 1e-6:  # No improvement
            _stall_count += 1
        else:
            _stall_count = 0
    _prev_f_best = f_best
    
    # Ensure sigma is non-negative
    sigma = np.abs(sigma)
    
    # Use Expected Improvement (EI) as the base acquisition
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    eps = 1e-8
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small UCB component for exploration, especially when stalled
    # The UCB weight increases with stalls to encourage exploration
    base_ucb_weight = 0.1
    stall_bonus = min(_stall_count * 0.15, 0.5)  # Max 0.5 extra
    ucb_weight = base_ucb_weight + stall_bonus
    
    # Normalize sigma to [0, 1] range for consistent UCB scaling
    sigma_max = np.max(sigma) if np.max(sigma) > eps else 1.0
    sigma_norm = sigma / sigma_max
    
    ucb_component = ucb_weight * sigma_norm
    
    # Combine EI and UCB
    scores = ei + ucb_component
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
