import numpy as np
import time
import scipy.special

# Global state
_start_time = None

# Weibull distribution parameters
# Based on successful programs, using k=2.0 and lambda=15.0
# Mean = lambda * Gamma(1 + 1/k) = 15 * Gamma(1.5) = 15 * 0.8862 = 13.29
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0
_EXPECTED_NEXT_ITEM = _WEIBULL_LAMBDA * scipy.special.gamma(1 + 1/_WEIBULL_K)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Calculate the survival function S(r) = exp(-(r/lambda)^k)
    # This estimates the probability that a remainder r can be filled by future items
    r = new_rems
    arg = (r / _WEIBULL_LAMBDA) ** _WEIBULL_K
    # Clip arg to prevent underflow issues in exp
    arg = np.clip(arg, 0, 100)
    S_r = np.exp(-arg)
    
    # Dead Space term: r * S(r)
    # This represents the expected dead space if the remainder is not filled
    dead_space = r * S_r
    
    # Linear consolidation penalty
    # Encourages using bins with smaller remainders (Best Fit behavior)
    alpha = 0.1
    linear_penalty = alpha * r
    
    # Polarization penalty from Inspiration 2
    # Penalizes "middle" remainders (around CAPACITY/2)
    # min(rem_after, CAPACITY - rem_after) measures "how close to the middle"
    pol_penalty = np.minimum(new_rems, CAPACITY - new_rems)
    pol_weight = 0.5
    
    # Total penalty
    penalty = dead_space + linear_penalty + pol_weight * pol_penalty
    
    # Score is negative penalty (we want to minimize penalty)
    final_scores = -penalty
    
    # Add a tiny index penalty for tie-breaking (lower index preferred)
    final_scores -= np.arange(len(valid_indices)) * 1e-9
    
    scores[valid] = final_scores
    
    return scores
