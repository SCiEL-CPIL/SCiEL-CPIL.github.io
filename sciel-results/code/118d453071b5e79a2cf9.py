"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull distribution parameters for the generated sequences
# The task mentions "five generated Weibull sequences" with capacity 100.
# A common setup for these tasks uses Weibull(k, lambda) where items fit into bins of size 100.
# Assuming a standard Weibull with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and median of ~19, which is reasonable for capacity 100.
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

# Mode of the Weibull distribution
# For k > 1, mode = lambda * ((k-1)/k)^(1/k)
_MODE = _lambda_scale * (( _k - 1) / _k) ** (1 / _k)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Inventory System Modeling with Mode-Targeting and Dead-Space Penalty.
    
    We model the bin packing process as a discrete-time inventory system.
    The goal is to steer the remaining capacity (inventory) of each bin towards
    the most probable future item size (the mode of the Weibull distribution).
    By doing so, we maximize the likelihood that the next arriving item fits
    perfectly (or very well), minimizing fragmentation and waste.
    
    Additionally, we incorporate a "Dead-Space" penalty inspired by the
    probabilistic best-fit approach. This penalty discourages remainders that
    are too small to be filled by future items (high probability of being
    wasted) or too large (wasting capacity that could be used elsewhere).
    
    Score components:
    1. Mode-Targeting: - (gap - Mode)^2 / sigma_mode
       - Encourages gaps close to the mode of the item size distribution.
       - This aligns the remaining space with the most likely next item size.
       
    2. Dead-Space Penalty: - (gap * S(gap))
       - S(gap) is the survival function of the Weibull distribution.
       - S(gap) = exp(-(gap/lambda)^k)
       - This term penalizes remainders that are likely to remain unfilled.
       
    3. Linear Consolidation: - alpha * gap
       - A small linear penalty to encourage best-fit behavior and consolidation.
    
    Final Score = Mode-Targeting Score + Dead-Space Penalty + Linear Consolidation
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    feasible_gaps = remaining_after[feasible]
    
    # 1. Mode-Targeting Score
    # We want to minimize the squared deviation from the mode.
    # Score contribution: - (gap - Mode)^2 / sigma_mode
    # sigma_mode controls the sharpness of the preference for the mode.
    # A smaller sigma_mode makes the preference sharper.
    sigma_mode = 10.0
    mode_score = -((feasible_gaps - _MODE) ** 2) / sigma_mode
    
    # 2. Dead-Space Penalty
    # Calculate the survival function S(r) = exp(-(r/lambda)^k)
    r = feasible_gaps
    arg = (r / _lambda_scale) ** _k
    # Clip arg to prevent underflow issues in exp, though exp(-1000) is 0 anyway.
    S_r = np.exp(-arg)
    dead_space_penalty = -(r * S_r)
    
    # 3. Linear Consolidation Penalty
    alpha = 0.1
    linear_penalty = -alpha * r
    
    # Combine scores
    final_scores = mode_score + dead_space_penalty + linear_penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
