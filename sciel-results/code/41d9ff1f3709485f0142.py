"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for adaptive exploration
_start_time = None
_prev_f_best = None
_stagnation_count = 0
_explore_weight = 0.5  # starts at 0.5, increases with stagnation


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that blends Kriging Believer (probabilistic improvement)
    with an exploration term, adaptively weighted based on stagnation.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    global _start_time, _prev_f_best, _stagnation_count, _explore_weight
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Time guard: if we're getting close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 320:  # 10s safety margin
        # Fallback: return scores based on expected improvement per unit cost
        # This is deterministic and fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / (cost + 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Detect stagnation: if f_best hasn't improved significantly, increase exploration
    if _prev_f_best is not None:
        if f_best >= _prev_f_best - 1e-6:  # no improvement
            _stagnation_count += 1
        else:
            _stagnation_count = 0
    _prev_f_best = f_best
    
    # Adaptively adjust exploration weight
    # More stagnation -> more exploration
    _explore_weight = min(1.0, 0.3 + 0.1 * _stagnation_count)
    
    # Compute Kriging Believer score: probability of improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / (sigma + 1e-9)
    p_improve = norm.cdf(z)
    
    # Exploration term: use the GP mean shifted by uncertainty
    # We want to explore where sigma is large, but also where mu is promising
    # Use a UCB-like term for exploration
    beta = 2.0  # exploration coefficient
    explore_score = -mu + beta * sigma  # higher is better for exploration
    
    # Normalize both scores to [0, 1] for blending
    # For p_improve, it's already in [0, 1]
    # For explore_score, normalize
    explore_min = np.min(explore_score)
    explore_max = np.max(explore_score)
    if explore_max - explore_min > 1e-9:
        explore_norm = (explore_score - explore_min) / (explore_max - explore_min)
    else:
        explore_norm = np.ones_like(explore_score)
    
    # Blend: (1 - w) * exploitation + w * exploration
    blended = (1 - _explore_weight) * p_improve + _explore_weight * explore_norm
    
    # Divide by cost to get cost-aware score
    scores = blended / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
