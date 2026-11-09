"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the bin with the highest score.
    
    Strategy: Soft best-fit with a smooth penalty for waste.
    We want to minimize bins used (L1 lower bound = sum(items)/capacity).
    Best-fit reduces waste per bin, but we smooth the selection to avoid
    pathological tie-breaking and use a temperature that favors tighter fits
    while still considering all feasible bins.
    """
    # Feasibility: only bins with enough remaining capacity
    remaining = bins - item
    feasible = remaining >= 0
    
    # If no feasible bins, return all -inf (should not happen per problem setup)
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # For feasible bins, score based on how tightly they fit
    # We want to prefer bins where remaining capacity after placing item is small,
    # but use a smooth function to avoid hard min behavior
    
    # Work with log-sum-exp style soft-min on remaining capacity
    # Lower remaining = better fit
    
    # Mask out infeasible bins by setting them to a large positive value
    # so they get very low scores
    masked_remaining = np.where(feasible, remaining, 1e9)
    
    # Soft-min: score = -softmin(remaining)
    # softmin(x) = -logsumexp(-x) for minimum
    # We want to minimize remaining, so we compute -logsumexp(-masked_remaining)
    
    # Scale factor: temperature for the soft-min
    # Smaller tau = sharper (closer to hard min)
    # Larger tau = smoother
    # Capacity is 100, so remaining is in [0, 100]
    tau = 1.0
    
    # Compute soft-min of remaining capacities for feasible bins
    # For numerical stability, subtract max before exp
    neg_masked = -masked_remaining / tau
    max_neg = np.max(neg_masked)
    exp_neg = np.exp(neg_masked - max_neg)
    log_sum_exp = max_neg + np.log(np.sum(exp_neg))
    
    # softmin = -logsumexp(-x) = -(max_neg + log(sum(exp(neg_masked - max_neg))))
    # But we want the score to be higher for tighter fits
    # score = -softmin(remaining) = logsumexp(-remaining/tau) + max_neg... 
    # Actually, let's just use: score = -softmin(remaining)
    # softmin(remaining) ≈ min(remaining) when tau is small
    # So score ≈ -min(remaining), which is best-fit
    
    soft_min_value = -(max_neg + np.log(np.sum(exp_neg)))
    score = -soft_min_value
    
    # For infeasible bins, set score to -inf
    score = np.where(feasible, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
