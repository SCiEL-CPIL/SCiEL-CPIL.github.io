"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute a lookup table for the scoring function to avoid per-call overhead.
# We use a "gap penalty" approach: score = -f(remaining_gap)
# where f is a convex function that penalizes large gaps but with diminishing sensitivity
# for very small gaps (to avoid over-penalizing tiny gaps that don't matter much).

# We'll use a piecewise function or a smooth approximation.
# Let's use: score = -gap - epsilon * (gap^2) for small gaps, but this is just quadratic.
# A better structural change: use a "fill ratio" based score with a non-linear transform.
# Let remaining = bins - item.
# fill_ratio = 1 - remaining/100.
# We want to maximize fill_ratio, but with a non-linear boost for high fill ratios.
# Let's try: score = fill_ratio + lambda * (fill_ratio^2)
# This encourages bins that are already quite full to get even fuller.

# Pre-compute a table for the score based on the remaining capacity.
# Since remaining can be any float in [0, 100], we can't easily table it.
# But we can use a fast analytical function.

# Let's try a different structural approach: 
# Instead of just penalizing remaining space, we can look at the "waste" created.
# Waste = remaining_after_placement.
# We want to minimize waste. 
# But pure Best Fit (minimize waste) is already a strong baseline.
# A structural change could be to consider the "future" impact.
# For example, if placing an item leaves a gap that is too small for any future item, 
# it's "dead" space. But we don't know future items.
# However, we can estimate that gaps smaller than some threshold are "dead".
# This is similar to "Best Fit Decreasing" but online.

# Let's try a "gap penalty" with a non-linear function that is very steep for small gaps
# and less steep for large gaps. This encourages filling bins to near capacity.
# score = -gap / (gap + k)  ? No, this is always negative and approaches -1.
# Let's use: score = -gap * (1 + a * exp(-gap/b))
# This penalizes large gaps more heavily due to the exponential term.

# Let's simplify: use a power law on the remaining capacity.
# score = -(remaining)^alpha
# For alpha < 1, this is less sensitive to large gaps and more sensitive to small gaps?
# No, if alpha < 1, the function is concave. 
# If alpha > 1, the function is convex.
# We want to prefer small gaps. So we want a function that is very low (bad) for large gaps
# and less low (good) for small gaps.
# So we want a convex function of remaining.
# score = -remaining^alpha with alpha > 1.
# Let's try alpha = 1.5 or 2.0.

# But the current program already uses a power law with alpha=2.0.
# Let's try a different structural change: 
# Use a "hybrid" of Best Fit and Worst Fit.
# Best Fit: minimize remaining.
# Worst Fit: maximize remaining (to keep bins open for large future items).
# A common heuristic is "Best Fit Decreasing" but we don't have future items.
# We can use a "weighted" combination: score = -w * remaining + (1-w) * remaining
# This is just linear.
# Let's try: score = -remaining + c * remaining^2
# This is a quadratic that penalizes large gaps more.

# Let's try a "fill ratio" based score with a non-linear transform.
# fill_ratio = (100 - remaining) / 100
# score = fill_ratio^2 + 0.1 * fill_ratio
# This encourages high fill ratios.

# Let's pre-compute a table for the score based on the remaining capacity.
# We'll use a lookup table for speed.
# Remaining capacity can be from 0 to 100.
# We'll use a table of size 10000 (0.01 resolution).

_TABLE_SIZE = 10000
_table = np.zeros(_TABLE_SIZE, dtype=np.float64)
for i in range(_TABLE_SIZE):
    remaining = i * 0.01  # 0.0 to 100.0
    fill_ratio = 1.0 - remaining / 100.0
    # Non-linear score: prefer high fill ratios
    # Use a power law on fill_ratio to emphasize high fill ratios
    score = np.power(fill_ratio, 2.0) + 0.05 * fill_ratio
    _table[i] = score

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a pre-computed lookup table for speed.
    The score is based on the fill ratio after placing the item.
    """
    start_time = time.time()
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 140:
        return scores
    
    # Use the lookup table to compute scores
    # remaining is in [0, 100], so we can index into the table
    # Table index = int(remaining / 0.01)
    # Clamp to valid range
    indices = np.clip((remaining[valid_mask] * 100).astype(np.int64), 0, _TABLE_SIZE - 1)
    scores[valid_mask] = _table[indices]
    
    return scores
# EVOLVE-BLOCK-END
