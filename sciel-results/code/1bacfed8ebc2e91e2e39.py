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

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Hybrid Dead-Space + Polarization Scoring.
    
    We combine the structural insight from the current program (Weibull Dead Space)
    with the refinement machinery from the inspirations (Polarization).
    
    1. Dead Space Penalty: r * S(r) where S(r) is the Weibull survival function.
       This penalizes remainders that are large enough to be "dead" (unlikely to be filled
       by future items) but not so large that they represent a significant portion of the bin.
       It creates a valley at r=0 and r->infinity, discouraging the "middle" dead zone.
       
    2. Polarization Penalty: min(r, CAPACITY - r).
       This explicitly penalizes remainders that are close to half the bin capacity,
       as these bins are less flexible for future packing.
       
    3. Linear Consolidation Penalty: alpha * r.
       A small linear term to encourage Best-Fit behavior (minimizing immediate waste).
       
    The final score is the negative of the sum of these penalties.
    Score = - (dead_space + pol_weight * pol_penalty + alpha * r)
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
    r = remaining_after[feasible]
    
    # 1. Calculate the survival function S(r) = exp(-(r/lambda)^k)
    # Clip r to avoid numerical issues, though r >= 0
    arg = (r / _lambda_scale) ** _k
    # Clip arg to prevent underflow issues in exp, though exp(-1000) is 0 anyway.
    S_r = np.exp(-np.clip(arg, 0, 100))  # Clip to [0, 100] for safety
    
    # Dead Space term: r * S(r)
    dead_space = r * S_r
    
    # 2. Polarization penalty (from Inspirations)
    # min(r, CAPACITY - r) measures "how close to the middle"
    # We want to MINIMIZE this, so we add it to the penalty
    pol_penalty = np.minimum(r, CAPACITY - r)
    
    # Weight for polarization penalty
    # For r ~ 20, pol_penalty ~ 20. dead_space ~ 20 * exp(-(20/25)^2) ~ 20 * 0.45 ~ 9.
    # We want pol_penalty to be comparable to dead_space, so weight ~ 0.5
    pol_weight = 0.5
    
    # 3. Linear consolidation penalty
    alpha = 0.1
    linear_penalty = alpha * r
    
    # Total penalty
    penalty = dead_space + pol_weight * pol_penalty + linear_penalty
    
    # Score is negative penalty (we want to minimize penalty)
    final_scores = -penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
