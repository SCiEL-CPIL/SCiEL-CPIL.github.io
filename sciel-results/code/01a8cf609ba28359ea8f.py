"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

_START_TIME = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

# Weibull distribution parameters for the generated sequences
_k = 2.0
_lambda_scale = 25.0

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback: Best Fit (minimize remaining)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Adversarial Robustness (Worst Fit Variant)
    #
    # The directive suggests treating the process as a game-theoretic adversarial
    # robustness problem. We prioritize bins that create remainders robust against
    # the worst-case immediate future demand (i.e., large items).
    #
    # This translates to maximizing the remaining capacity (Worst Fit), which
    # preserves the ability to accommodate large items that may arrive next.
    # Unlike Best Fit, which minimizes immediate waste, Worst Fit minimizes the
    # risk of being unable to place a future large item, which is a higher-cost
    # event in online packing.
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    feasible_gaps = remaining_after[feasible]
    
    # Base Score: Worst Fit (Maximize remaining space)
    # We want to maximize the gap. So base score is +gap.
    base_score = feasible_gaps
    
    # Robustness Adjustment:
    # While we favor large remainders, we apply a slight penalty for remainders
    # that are "too large" to be efficiently utilized by the *typical* next item,
    # balancing the adversarial robustness with average-case efficiency.
    # However, the primary driver is the Worst Fit signal.
    # Let's use a simple Worst Fit for this structural change.
    
    final_scores = base_score
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
