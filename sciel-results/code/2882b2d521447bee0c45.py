import numpy as np
import time

_start_time = None
DEADLINE = 300.0
SAFETY = 20.0

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    # Ensure sigma is strictly positive to avoid division issues if used in other variants, 
    # though for UCB we just multiply.
    s = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Deadline guard
    if _elapsed() > DEADLINE - SAFETY:
        # Fallback: Pure Exploitation (argmin mu)
        return -mu
    
    # Adaptive UCB
    # f_best is the current best (lowest) value.
    # The global minimum is 0.
    # As f_best approaches 0, we should reduce exploration (beta).
    
    # Sanitize f_best
    fb = float(f_best)
    if not np.isfinite(fb):
        fb = 0.0
    
    # If f_best is negative (should not happen if min is 0, but robustness), 
    # treat it as 0 for the exploration decay logic, or even increase exploitation.
    # If f_best < 0, we found something better than the minimum? Unlikely.
    # We will clamp fb to [0, infinity) for the beta calculation.
    fb_clamped = max(0.0, fb)
    
    # Beta parameters
    beta_min = 0.1
    beta_max = 5.0
    
    # Decay function: starts at beta_max when f_best is large, 
    # approaches beta_min as f_best -> 0.
    # Using exponential decay.
    # exp(-f_best) is 1 at 0, 0 at infinity.
    # 1 - exp(-f_best) is 0 at 0, 1 at infinity.
    # We want beta to be HIGH when f_best is LARGE (early).
    # So beta = beta_min + (beta_max - beta_min) * (1 - exp(-f_best))
    # Check:
    # f_best = 0 -> 1 - 1 = 0 -> beta = beta_min. (Correct, exploit)
    # f_best = 10 -> 1 - ~0 = 1 -> beta = beta_max. (Correct, explore)
    
    decay_factor = 1.0 - np.exp(-fb_clamped)
    beta = beta_min + (beta_max - beta_min) * decay_factor
    
    # Calculate UCB scores for minimization
    # Score = -mu + beta * sigma
    # Higher score is better.
    scores = -mu + beta * s
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
