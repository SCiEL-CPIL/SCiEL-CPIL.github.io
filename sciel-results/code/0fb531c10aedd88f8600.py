# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

_call_count = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _call_count, _last_f_best
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe
    
    # Reset logic for new seed/run
    if _last_f_best is not None and f_best > _last_f_best:
        _call_count = 0
    _last_f_best = f_best
    _call_count += 1
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw a single sample from the posterior
    # For minimization, we want to sample from -f (maximize) or equivalently
    # score = -y_sample where y_sample ~ N(mu, sigma^2)
    # This naturally balances exploration and exploitation
    
    # Draw one random sample per candidate from the posterior
    rng = np.random.default_rng()
    eps = rng.standard_normal(n)
    y_sample = mu + sigma_safe * eps
    
    # Score is negative of the sample (higher = better for minimization)
    score = -y_sample
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
