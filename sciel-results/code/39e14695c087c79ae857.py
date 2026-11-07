"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_global_start_time = None
_global_call_count = 0

def acquisition(mu, sigma, f_best):
    global _global_start_time, _global_call_count
    
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Deadline guard: 330s total, keep safety margin
    elapsed = time.time() - _global_start_time
    if elapsed > 310:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Determine the call index (0-indexed)
    # The first call to acquisition() is for the 6th evaluation.
    # There are 25 calls in total (evaluations 6 through 30).
    current_call_idx = _global_call_count
    _global_call_count += 1
    
    total_calls = 25
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add an adaptive exploration bonus that decays over the run
    # This helps explore the multimodal Branin landscape early on
    # The bonus is proportional to sigma, encouraging high-uncertainty regions
    # Decay from a strong early bonus to a small late bonus
    bonus_start = 0.8
    bonus_end = 0.1
    
    if total_calls > 1:
        progress = current_call_idx / (total_calls - 1)
        # Use a smooth decay (quadratic) to keep exploration meaningful longer
        bonus = bonus_start - (bonus_start - bonus_end) * (progress ** 1.5)
    else:
        bonus = bonus_start
        
    # Combine EI with exploration bonus
    scores = ei_scores + bonus * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
