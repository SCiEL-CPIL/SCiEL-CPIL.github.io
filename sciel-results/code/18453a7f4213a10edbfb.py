"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We maintain a small cache to avoid recomputing expensive stats if possible,
# but since bins change every call, we must compute fast.
# The key insight for online bin packing with Weibull distributions (typically heavy-tailed or skewed)
# is that Best-Fit is strong, but we can improve by considering the "shape" of the remaining space.
# A common improvement is "Best Fit Decreasing" but we are online.
# Another is to prefer bins that have a remaining capacity close to the item size,
# but also penalize bins that are "fragmented" or have very small remaining space that won't fit future items.
# However, we don't know future items.

# Let's look at the L1 lower bound aspect.
# The score is mean percent excess of bins used over the L1 lower bound.
# L1 lower bound is usually sum(items) / capacity.
# To minimize bins used, we want to pack tightly.

# Strategy:
# 1. Filter bins with enough capacity.
# 2. Among those, prefer the bin that minimizes the remaining space after placing the item (Best Fit).
# 3. Tie-breaker: If multiple bins have the same remaining space, prefer the one with larger total capacity? No, capacity is fixed (100).
# 4. Alternative: "Worst Fit" is often bad for minimizing bin count.
# 5. "Next Fit" is bad.
# 6. "First Fit" is bad.

# Let's try a hybrid:
# Score = - (bins - item)  # Best Fit (maximize score means minimize remaining)
# But wait, the function returns scores, and the item goes to the HIGHEST scoring bin.
# So we want high score for good bins.
# Best Fit: remaining = bins - item. We want small remaining. So score = -remaining.
# This is exactly what the seed program does: return -(bins - item).

# Why is the seed program score 1.0? It's the baseline.
# Can we do better?
# Maybe "Best Fit" is not optimal for Weibull.
# Weibull distributions can be skewed. If shape parameter k < 1, there are many small items and few large ones.
# If k > 1, more uniform.
# Usually, for bin packing, Best Fit is very strong.

# Let's consider "Best Fit with a penalty for very small remaining spaces".
# If a bin has 1.0 remaining and item is 0.5, remaining becomes 0.5.
# Future items of size > 0.5 won't fit.
# If we have another bin with 2.0 remaining, and item is 0.5, remaining becomes 1.5.
# Future items up to 1.5 can fit.
# So maybe we should prefer bins that leave a "usable" remainder.

# Heuristic: Score = - (bins - item) + alpha * (bins - item) ? No.
# Let's try: Score = - (bins - item) if (bins - item) < threshold else something else.

# Actually, a known strong heuristic for online bin packing is "Best Fit".
# Another is "Worst Fit" which is bad.
# "First Fit Decreasing" is offline.

# Let's look at the "percent excess" metric.
# Excess = (BinsUsed - LB) / LB.
# Minimizing BinsUsed is the goal.

# Perhaps we can use a "Lookahead" style heuristic without actual lookahead?
# Estimate the distribution of remaining items.
# If we assume items are Weibull, we can estimate the probability that a future item fits in the remaining space.
# But we don't know the parameters of the Weibull distribution.

# Let's try a simple improvement:
# If the remaining space in the best-fit bin is very small (e.g., < 5% of capacity),
# it might be better to put the item in a fuller bin (less remaining) or a new bin?
# No, putting in a new bin increases bin count immediately.
# Putting in a bin with more remaining space might save a bin later?

# Let's try "Best Fit" but with a tie-breaker or slight modification.
# What if we penalize bins that are almost full but not full?
# No, that's what Best Fit does.

# Let's try "Worst Fit" for a moment? No, that's usually worst.

# How about "Best Fit" but if the best fit leaves less than X remaining,
# and there is a bin that leaves more remaining, prefer the one with more remaining?
# This is "Worst Fit" among the "Best Fit" candidates? No.

# Let's try a score that balances tightness and flexibility.
# Score = - (bins - item) + beta * (bins - item)^2 ?
# This penalizes large remainders more heavily? No, quadratic grows faster.
# If we want to prefer small remainders, -x is good.
# If we want to avoid extremely small remainders (wasted space), we might add a term.

# Let's stick to a robust heuristic:
# 1. Calculate remaining = bins - item.
# 2. Filter remaining >= 0.
# 3. Score = -remaining.
# 4. But if remaining is very small (e.g., < 1.0), it's likely wasted.
#    Maybe we should prefer a bin with slightly more remaining if it's not too much?
#    Actually, Best Fit is proven to be within 1.7 of optimal for online bin packing.
#    It's hard to beat it significantly without knowing the distribution.

# However, the task mentions "Weibull sequences".
# Weibull with shape k=1 is Exponential.
# For Exponential items, Best Fit is good.
# Is there a better heuristic for Exponential?
# Some papers suggest "Best Fit" is near-optimal.

# Let's try a different angle: The L1 lower bound.
# The score is excess over L1

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Hybrid Best-Fit/Worst-Fit with fragmentation penalty.
    
    Key insights:
    1. Pure Best-Fit minimizes immediate waste but can create many small unusable gaps.
    2. Pure Worst-Fit preserves large spaces but wastes capacity on small items.
    3. For Weibull (skewed) distributions, we need to balance:
       - Prefer tight fits (Best-Fit) to maximize density.
       - Avoid creating tiny gaps that will never be filled (fragmentation penalty).
       - Slightly prefer bins with more remaining space if the tight fit creates a "dead" gap.
    
    Scoring formula:
    - Base: -waste (Best-Fit component)
    - Bonus: alpha * bins[i] (Worst-Fit component, preserves capacity)
    - Penalty: beta * waste * (1 - waste/item) for small gaps (waste < item)
    
    The penalty term is designed to discourage creating gaps smaller than the current item,
    as these are likely unusable for future items of similar scale.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    if not np.any(valid):
        return scores
        
    waste = remaining[valid]
    bin_rem = bins[valid]
    
    # Base score: Best-Fit (minimize waste)
    base_score = -waste
    
    # Worst-Fit component: slightly prefer bins with more remaining capacity
    # This helps preserve large contiguous space for future large items
    # We use a small coefficient to not override the Best-Fit signal
    worst_fit_bonus = 0.05 * bin_rem
    
    # Fragmentation penalty: penalize creating small gaps that are likely unusable
    # If waste is small relative to the item, it's probably dead space
    # We use a smooth penalty that is 0 when waste >= item and increases as waste -> 0
    # Penalty = beta * waste * max(0, 1 - waste/item)
    # This is 0 when waste >= item, and quadratic-like when waste < item
    
    # To avoid division by zero, we add a small epsilon
    item_safe = item + 1e-9
    gap_ratio = waste / item_safe
    # Penalty is active when gap_ratio < 1 (waste < item)
    # We use a smooth transition
    penalty_mask = gap_ratio < 1.0
    penalty = np.zeros_like(waste)
    if np.any(penalty_mask):
        # For small gaps, penalize based on how small they are
        # A gap of 0 is worst, a gap of item is fine
        # Penalty = beta * waste * (1 - gap_ratio)
        # This gives 0 penalty at waste=item and beta*waste penalty at waste=0
        # But waste=0 is actually perfect (exact fit), so we need to be careful
        # Actually, if waste is very small, it's "dead space" unless it's exactly 0
        # Let's use a penalty that peaks at some intermediate small value
        # Or simply: if waste < threshold, add a penalty
        # Let's use: penalty = 0.1 * waste * (1 - gap_ratio) for gap_ratio < 1
        # This encourages either exact fits (waste~0) or larger gaps (waste>item)
        # Wait, if waste is 0, penalty is 0. If waste is 0.5*item, penalty is 0.1*0.5*item*0.5 = 0.025*item
        # If waste is 1.5*item, penalty is 0.
        # This seems reasonable.
        penalty[penalty_mask] = 0.1 * waste[penalty_mask] * (1 - gap_ratio[penalty_mask])
    
    scores[valid] = base_score + worst_fit_bonus - penalty
    
    return scores

# EVOLVE-BLOCK-END
