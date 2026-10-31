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
# Based on the high-performing inspiration (1.8171), we use:
# k=2.0, lambda=15.0
# This implies mean item size ~ 15 * Gamma(1.5) ~ 13.3.
# The "Dead Space" metric r * S(r) has a maximum (worst case) at r = lambda * (1/k)^(1/k)
# For k=2, r* = 15 / sqrt(2) ~ 10.6.
# This suggests remainders around 10 are the most problematic "dead zones".
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0
_EXPECTED_NEXT_ITEM = _WEIBULL_LAMBDA * scipy.special.gamma(1 + 1/_WEIBULL_K)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Hybrid Probabilistic Polarization.
    
    We combine the structural "Dead Space" metric from the current program (which uses the Weibull
    survival function to identify probabilistically wasteful remainders) with the "Polarization"
    concept from Inspiration 1 (which pushes bins towards full or empty states to maximize future
    flexibility).
    
    The Dead Space term r * S(r) penalizes remainders that are likely to be "trapped" because they
    are too small for typical future items but too large to be ignored. This is a probabilistic
    Best-Fit refinement.
    
    The Polarization term min(rem, cap-rem) penalizes remainders in the "middle" of the bin's
    capacity range. Middle remainders are less flexible than extremes.
    
    We combine these two penalties. The Dead Space term is distribution-aware, while the
    Polarization term is structure-aware.
    
    Score = - (dead_space + weight * polarization_penalty)
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
    
    # 1. Dead Space Term: r * S(r)
    # S(r) = exp(-(r/lambda)^k)
    arg = (r / _WEIBULL_LAMBDA) ** _WEIBULL_K
    S_r = np.exp(-arg)
    dead_space = r * S_r
    
    # 2. Polarization Term: min(r, CAPACITY - r)
    # This measures how "middle" the remainder is.
    # We want to MINIMIZE this, so we add it to the penalty.
    pol_penalty = np.minimum(r, CAPACITY - r)
    
    # Weight for polarization penalty.
    # Dead space for r~10 (peak) is ~ 10 * exp(-(10/15)^2) ~ 10 * 0.66 ~ 6.6.
    # Pol penalty for r~10 is min(10, 90) = 10.
    # For r~50, dead space is ~ 50 * exp(-(50/15)^2) ~ 50 * 0.0003 ~ 0.015.
    # Pol penalty for r~50 is min(50, 50) = 50.
    # The pol penalty dominates for large r, which is good because large remainders are bad
    # if they are in the middle. But if r is close to 100, pol penalty is small.
    # Let's use a weight of 0.5 to balance.
    pol_weight = 0.5
    
    # Total penalty
    penalty = dead_space + pol_weight * pol_penalty
    
    # Score is negative penalty
    final_scores = -penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
