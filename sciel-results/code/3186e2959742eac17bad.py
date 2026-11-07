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
    # If we're running out of time, just return a simple EI score
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
    
    # Adaptive Beta for Lower Confidence Bound (LCB)
    # We use LCB for minimization: score = -mu + beta * sigma
    # Higher beta encourages exploration (high sigma regions).
    # We start with high exploration to find multiple local minima of Branin,
    # and decay to lower beta to exploit the best region.
    
    # Based on inspiration, decay from 4.0 to 1.5
    beta_start = 4.0
    beta_end = 1.5
    
    if total_calls > 1:
        # Linear interpolation
        # k=0 -> beta_start
        # k=24 -> beta_end
        progress = current_call_idx / (total_calls - 1)
        beta = beta_start - (beta_start - beta_end) * progress
    else:
        beta = beta_start
        
    # Calculate scores
    # We want to maximize this score.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We select min LCB(x), which is equivalent to max (-mu(x) + beta * sigma(x))
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
