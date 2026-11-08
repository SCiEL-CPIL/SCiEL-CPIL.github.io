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
# The score is excess over L1.
# If we can reduce the number of bins by even 1, the score drops significantly.
# Best Fit is good at reducing bin count.

# Let's try "Best Fit" with a small randomization? No, deterministic is better for reproducibility and scoring.

# What if we use "Next Fit Decreasing" logic? We can't sort.

# Let's try "First Fit" with a twist:
# Prefer the first bin that fits, but if there are multiple, pick the one with the smallest remaining space (Best Fit).
# That's just Best Fit.

# Let's try "Worst Fit" to see if it's better for some distributions?
# Unlikely.

# Let's try a "Hybrid" approach:
# If the item is large (> 50% capacity), Best Fit is critical.
# If the item is small, maybe Worst Fit (to keep bins balanced) is better?
# No, Worst Fit is generally bad for bin count.

# Let's try "Best Fit" but with a penalty for bins that are "almost empty" (large remaining).
# Score = - (bins - item) - gamma * (bins - item)
# This is just (1+gamma) * -(bins - item). Same relative ordering.

# What if we change the metric?
# Maybe we should prefer bins that have a remaining capacity that is a multiple of the item size? No.

# Let's look at the "percent excess" again.
# If we use fewer bins, the excess is lower.
# Best Fit is the standard strong heuristic.

# Is there a bug in the seed program?
# `return -(bins - item)`
# If `bins` is an array of remaining capacities.
# `bins - item` is remaining after placing item.
# We want to minimize this.
# So we want to maximize `- (bins - item)`.
# This is correct.

# Maybe the issue is that Best Fit can get stuck in local minima?
# Or maybe for Weibull, "Worst Fit" is better?
# Let's try a simple experiment in our head:
# Items: 0.6, 0.6, 0.6, 0.6
# Capacity: 1.0
# Best Fit:
# 0.6 -> Bin 0 (rem 0.4)
# 0.6 -> Bin 1 (rem 0.4)
# 0.6 -> Bin 2 (rem 0.4)
# 0.6 -> Bin 3 (rem 0.4)
# Total 4 bins.
# Worst Fit:
# 0.6 -> Bin 0 (rem 0.4)
# 0.6 -> Bin 0 (rem 0.4) -> Wait, Worst Fit puts in bin with most remaining.
# Bin 0 rem 0.4. Bin 1 rem 1.0.
# 0.6 -> Bin 1 (rem 0.4)
# 0.6 -> Bin 2 (rem 0.4)
# 0.6 -> Bin 3 (rem 0.4)
# Total 4 bins.

# Items: 0.5, 0.5, 0.5, 0.5
# Best Fit:
# 0.5 -> Bin 0 (rem 0.5)
# 0.5 -> Bin 0 (rem 0.0)
# 0.5 -> Bin 1 (rem 0.5)
# 0.5 -> Bin 1 (rem 0.0)
# Total 2 bins.
# Worst Fit:
# 0.5 -> Bin 0 (rem 0.5)
# 0.5 -> Bin 1 (rem 0.5)  [Bin 1 has 1.0 rem, Bin 0 has 0.5 rem]
# 0.5 -> Bin 0 (rem 0.0)
# 0.5 -> Bin 1 (rem 0.0)
# Total 2 bins.

# Items: 0.4, 0.4, 0.4, 0.4, 0.4, 0.4
# Best Fit:
# 0.4 -> B0 (0.6)
# 0.4 -> B0 (0.2)
# 0.4 -> B1 (0.6)
# 0.4 -> B1 (0.2)
# 0.4 -> B2 (0.6)
# 0.4 -> B2 (0.2)
# Total 3 bins.
# Worst Fit:
# 0.4 -> B0 (0.6)
# 0.4 -> B1 (0.6)
# 0.4 -> B2 (0.6)
# 0.4 -> B0 (0.2)
# 0.4 -> B1 (0.2)
# 0.4 -> B2 (0.2)
# Total 3 bins.

# It seems Best Fit is robust.

# Let's try a "Smarter" Best Fit.
# What if we prefer bins that are already "open" (have some items) over empty bins?
# No, that's not how Best Fit works. Best Fit looks at remaining capacity.

# Let's try to incorporate the "Weibull" nature.
# If we assume the items are Weibull, we can estimate the mean item size.
# But we don't know the parameters.

# Let's try a simple modification:
# If the best-fit bin has a remaining capacity that is very small (e.g., < 0.1 * capacity),
# and there is another bin with a remaining capacity that is "moderate" (e.g., between 0.2 and 0.5 * capacity),
# maybe we should prefer the moderate one?
# This is risky.

# Let's stick with Best Fit but add a tie-breaker:
# If two bins have the same remaining capacity, prefer the one with the larger index? No.
# Prefer the one that was used more recently? No state.

# What if we use "First Fit" instead of "Best Fit"?
# First Fit is faster but usually worse.

# Let's try "Best Fit" with a small epsilon to break ties?
# No.

# Let's try a different heuristic: "Worst Fit" is sometimes used to avoid fragmentation in some contexts, but for minimizing bin count, Best Fit is better.

# Let's try "Best Fit" but if the remaining space is < item_size / 2, penalize it?
# No, if remaining space is < item_size, it doesn't fit.

# Let's try "Best Fit" but if the remaining space is very small, say < 5,
# and there is a bin with remaining space > 50, prefer the > 50 bin?
# This would be "Worst Fit" in that case.

# Let's try a "Adaptive" heuristic:
# Count the number of bins used so far? We don't have that state.
# We only have `bins` array.

# Let's assume `bins` is an array of remaining capacities.
# We can infer the number of bins from the length of the array?
# No, the array might contain bins with 0 remaining capacity?
# The problem says "bins with enough remaining capacity".
# So we filter `bins >= item`.

# Let's try to improve the score function.
# Score = - (bins - item)
# This is linear.
# What if Score = - log(bins - item + epsilon)?
# This penalizes small remainders more heavily?
# If rem = 0.1, log(0.1) = -2.3. Score = 2.3.
# If rem = 0.5, log(0.5) = -0.69. Score = 0.69.
# So it prefers smaller remainders even more strongly.
# This might be good.

# Let's try: Score = - np.log(bins - item + 1e-6)
# But we must ensure bins - item > 0.

# Let's try another one: Score = - (bins - item) / (bins)
# This normalizes by the bin's capacity?
# If bin capacity is 100, and rem is 10, score = -0.1.
# If rem is 1, score = -0.01.
# This prefers larger relative remainders? No, -0.1 < -0.01, so it prefers rem=1.
# This is still Best Fit.

# Let's try: Score = - (bins - item) + 0.1 * (bins - item)
# No.

# Let's try a "Hybrid" of Best Fit and Worst Fit.
# Score = - (bins - item) + 0.1 * (max(bins) - bins)
# This adds a small incentive to use bins that are already full (small max(bins) - bins)?
# No, max(bins) is the max remaining capacity.
# If we use a bin with small remaining, max(bins) - bins is large.
# So we add a positive term.
# This would make the score higher for bins with small remaining.
# This reinforces Best Fit.

# Let's try the opposite:
# Score = - (bins - item) - 0.1 * (max(bins) - bins)
# This penalizes bins with small remaining?
# No, if rem is small, max(bins) - bins is large, so we subtract a large number.
# This would make the score lower for bins with small remaining.
# This would prefer bins with larger remaining.
# This is moving towards Worst Fit.

# Let's try a "Balanced" approach:
# We want to minimize the number of bins.
# Best Fit is the standard.
# Let's try to implement Best Fit with a slight preference for bins that are "not too full" to avoid fragmentation?
# No, fragmentation is bad.

# Let's try "Best Fit" but if the best fit leaves < 10% of capacity,
# and there is a bin that leaves > 50% of capacity,
# prefer the > 50% bin?
# This is a heuristic.

# Let's code a simple version of Best Fit and see if we can tweak it.
# Since we can't run it, we have to guess.

# A known improvement for online bin packing is "Best Fit" with "Tie-breaking by bin index" (First Fit among ties).
# The seed program doesn't handle ties explicitly, but NumPy `argmax` picks the first one.
# So it's already Best Fit with First Fit tie-breaking.

# What if we use "Worst Fit" for small items and "Best Fit" for large items?
# If item < 20, use Worst Fit?
# If item > 20, use Best Fit?

# Let's try this hybrid.
# Threshold: 20 (20% of capacity 100).

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Filter bins with enough capacity
    valid = bins >= item
    if not np.any(valid):
        # This shouldn't happen if the evaluator ensures a bin is available,
        # but we must return something.
        # Return a very low score for all, or -inf?
        # The contract says "score each bin with enough remaining capacity".
        # If none, we return an array of scores.
        # Let's return -np.inf for invalid bins.
        scores = np.full_like(bins, -np.inf, dtype=float)
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    remaining[~valid] = np.inf  # Invalid bins have infinite remaining (bad)

    # Hybrid heuristic
    if item < 20:
        # Worst Fit: prefer bins with largest remaining capacity
        # Score = remaining
        scores = remaining.copy()
    else:
        # Best Fit: prefer bins with smallest remaining capacity
        # Score = -remaining
        scores = -remaining.copy()

    # Set invalid bins to very low score
    scores[~valid] = -np.inf

    return scores

# Wait, the seed program is Best Fit.
# If I change to Hybrid, it might be better or worse.
# Let's try a more subtle improvement.
# "Best Fit" is optimal for many distributions.
# Maybe the issue is that the evaluator uses Weibull, and Best Fit is not optimal for Weibull?
# Some studies show that "Worst Fit" can be better for certain skewed distributions.
# But Weibull is often used to model failure times, which can be skewed.

# Let's try "Worst Fit" as the main heuristic?
# No, that's risky.

# Let's try "Best Fit" but with a penalty for bins that are "almost empty" (large remaining).
# Score = - (bins - item) + 0.01 * (bins - item)
# This is just 0.99 * -(bins - item). Same ordering.

# Let's try: Score = - (bins - item) - 0.01 * (bins)
# This penalizes bins with larger capacity? No, capacity is fixed.

# Let's try: Score = - (bins - item) + 0.01 * (item)
# This is a constant offset. Same ordering.

# Let's try a non-linear score.
# Score = - np.sqrt(bins - item)
# This prefers small remainders, but less aggressively than linear.
# If rem = 1, score = -1.
# If rem = 4, score = -2.
# If rem = 9, score = -3.
# Linear: -1, -4, -9.
# So sqrt prefers larger remainders relative to linear.
# This is moving towards Worst Fit.

# Let's try: Score = - (bins - item) ** 0.5
# This might be better for Weibull?

# Let's try: Score = - (bins - item) ** 1.5
# This prefers small remainders more aggressively.

# Let's try a "Mixture" of Best Fit and Worst Fit.
# Score = -0.7 * (bins - item) + 0.3 * (max(bins) - bins)
# This is a compromise.

# Let's implement this Mixture.
# max(bins) is the max remaining capacity among valid bins.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)

    remaining = bins - item
    remaining[~valid] = np.inf

    # Get max remaining among valid bins
    valid_remaining = remaining[valid]
    max_rem = np.max(valid_remaining)

    # Mixture: 70% Best Fit, 30% Worst Fit
    # Best Fit score: -remaining
    # Worst Fit score: remaining (or max_rem - remaining? No, remaining is fine)
    # Let's use: Score = -0.7 * remaining + 0.3 * (max_rem - remaining)
    # This simplifies to: Score = -remaining + 0.3 * max_rem
    # The 0.3 * max_rem is a constant for all bins, so it doesn't change the ordering!
    # So it's just Best Fit.

    # To get a real mixture, we need a term that depends on the bin's position relative to others.
    # Score = -0.7 * remaining + 0.3 * (max_rem - remaining) is not a mixture in the sense of changing ordering.
    # We need: Score = -0.7 * remaining + 0.3 * remaining = -0.4 * remaining. Still Best Fit.

    # Let's use: Score = -0.7 * remaining + 0.3 * (bins)
    # Since bins = item + remaining, this is -0.7 * remaining + 0.3 * item + 0.3 * remaining
    # = -0.4 * remaining + 0.3 * item.
    # Still Best Fit.

    # To get a true mixture, we need to compare bins to each other.
    # Score = -0.7 * remaining + 0.3 * (mean(remaining) - remaining)
    # = -remaining + 0.3 * mean(remaining)
    # Still Best Fit.

    # It seems any linear combination of remaining and constants is Best Fit.
    # We need a non-linear function.

    # Let's try: Score = - remaining + 0.1 * (remaining ** 2)
    # This penalizes large remainders more heavily?
    # If rem = 1, score = -1 + 0.1 = -0.9.
    # If rem = 2, score = -2 + 0.4 = -1.6.
    # If rem = 3, score = -3 + 0.9 = -2.1.
    # Linear: -1, -2, -3.
    # So it prefers smaller remainders more aggressively.

    # Let's try: Score = - remaining + 0.1 * (max_rem - remaining) ** 2
    # This is getting complex.

    # Let's stick with Best Fit but try a different power.
    # Score = - (bins - item) ** 0.8
    # This is slightly less aggressive than linear.

    # Let's try: Score = - (bins - item) ** 1.2
    # This is more aggressive.

    # Let's try the 1.2 power.

    scores = -np.power(remaining, 1.2, where=valid, out=np.full_like(remaining, -np.inf))
    return scores

# EVOLVE-BLOCK-END
