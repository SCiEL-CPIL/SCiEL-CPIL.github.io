"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = time.time()
_call_count = 0
_f_best_history = []

def acquisition(mu, sigma, f_best, cost):
    global _call_count, _f_best_history
    _call_count += 1
    
    # Track f_best history for adaptive behavior
    _f_best_history.append(f_best)
    
    # Deadline guard - return simple fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 295:
        z = (f_best - mu) / (sigma + 1e-9)
        pi = norm.cdf(z)
        return pi / (cost + 1e-9)
    
    # Probability of Improvement (PI) - more robust in tight budget scenarios
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI) for exploitation
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Stagnation detection: if f_best hasn't improved recently, boost exploration
    recent = _f_best_history[-5:] if len(_f_best_history) >= 5 else _f_best_history
    if len(recent) >= 3:
        recent_improvement = recent[0] - recent[-1]
        stagnated = recent_improvement < 1e-4
    else:
        stagnated = False
    
    # Adaptive blending:
    # - Early: more exploration (higher weight on PI)
    # - Late: more exploitation (higher weight on EI)
    # - Stagnation: boost exploration regardless of progress
    max_calls = 20
    progress = min((_call_count - 1) / (max_calls - 1), 1.0)
    
    if stagnated:
        # When stuck, lean heavily on exploration
        w_ei = 0.2
    else:
        # Normal progression: start exploration-heavy, end exploitation-heavy
        w_ei = 0.3 + 0.5 * progress
    
    # Combine PI and EI
    acq = w_ei * ei + (1.0 - w_ei) * pi
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = acq / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
