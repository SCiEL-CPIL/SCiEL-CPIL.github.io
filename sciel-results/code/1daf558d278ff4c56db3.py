"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computed constants for Weibull sequences with shape k and scale lambda
# The task uses 5 generated Weibull sequences of 1000 items with capacity 100.
# Typical Weibull parameters for such tasks: shape k ~ 1-3, scale lambda such that mean item size is reasonable.
# Since we don't know exact parameters, we use a robust heuristic.

# L1 lower bound: sum(items) / capacity
# We want to minimize: (num_bins_used - L1_bound) / L1_bound * 100
# Equivalently, minimize num_bins_used (since L1_bound is fixed for a given sequence)
# But the score is relative excess over L1 bound.

# Best-fit decreasing is strong, but we process online.
# Key insight: for online bin packing, best-fit (smallest feasible bin) is often near-optimal.
# The seed uses best-fit: return -(bins - item) which maximizes (bins - item), i.e., picks bin with least remaining space after placing item.

# Improvement ideas:
# 1. First-fit decreasing isn't possible online.
# 2. Best-fit is good. Can we improve with a slight modification?
# 3. Consider the distribution: Weibull items. If items are small, many fit per bin.
# 4. Try: prefer bins that, after placing item, leave remaining capacity closest to some target.
# 5. Or: use a score that considers both remaining capacity and bin utilization.

# Let's try a weighted score:
# score = -remaining_after + alpha * utilization_before
# where utilization_before = (capacity - bins[i]) / capacity
# This slightly prefers more-utilized bins, which can help pack tighter.

# Actually, best-fit already implicitly does this. Let's try a different angle:
# For Weibull distributions, there may be many small items. Best-fit handles this well.
# Try: combine best-fit with a small preference for bins that have moderate remaining space
# (not too much, not too little) to avoid fragmentation.

# Alternative: use the actual remaining capacity but with a non-linear transform.
# score = -(bins - item) is linear. Try:
# score = -log(bins - item + eps) for bins that can fit the item
# This prefers bins with very little remaining space more strongly.

# Let's implement a robust version:
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin. Higher score = preferred.
    bins[i] = remaining capacity of bin i.
    Only bins with bins[i] >= item can accept the item.
    """
    # Mask of bins that can fit the item
    feasible = bins >= item
    
    # For feasible bins, compute remaining space after placing item
    remaining_after = bins - item  # This is >= 0 for feasible bins
    
    # Best-fit: minimize remaining_after
    # Use negative remaining_after as score, but handle infeasible bins
    # For infeasible bins, assign a very low score (negative infinity effectively)
    
    # Use a large negative number for infeasible bins
    # For feasible bins, use -remaining_after (best-fit)
    
    # To make it more robust, add a small secondary criterion:
    # Among bins with similar remaining_after, prefer those with higher original capacity usage
    # i.e., prefer bins that were already more full
    
    # Simple approach: just best-fit
    scores = np.full_like(bins, -1e18, dtype=float)
    scores[feasible] = -remaining_after[feasible]
    
    return scores
# EVOLVE-BLOCK-END
