import numpy as np

# We assume capacity is 100 as per task description
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    # For valid bins, we want to select the one that minimizes waste.
    # Waste = remaining space after placing the item.
    # We want the highest score to correspond to the minimum remaining space.
    # So, score = -remaining.
    # This is the Best-Fit Decreasing (BFD) heuristic, which is generally
    # one of the best simple heuristics for bin packing.
    
    # However, the current program uses a threshold. Let's see if pure Best-Fit
    # or a refined version works better.
    # The inspiration 1 (score 0.9900) is worse than current (0.9998).
    # The current program uses Worst-Fit for large items and Best-Fit for small items.
    # Large items (> 50) must go in a new bin or a bin with very little space? No, if item > 50, it needs > 50 space.
    # If item is 60, remaining must be >= 60.
    # Worst-Fit for large items: maximize remaining. This spreads load.
    # Best-Fit for small items: minimize remaining. This packs tight.
    
    # Let's try to refine the threshold or the scoring.
    # A common improvement is "First Fit Decreasing" but we don't have order.
    # "Best Fit Decreasing" is standard.
    
    # Let's look at the "Next-Item Awareness" hint.
    # Estimate expected size of next item.
    # We can estimate the distribution of items from the current bins?
    # The bins contain the sum of items placed in them.
    # But we don't have the individual items.
    # However, the problem says "Weibull sequences". Weibull is skewed.
    # Most items are small, few are large.
    
    # If we assume the next item is drawn from the same distribution,
    # and we know the capacity, maybe we can estimate the mean item size.
    # But we don't know the distribution parameters.
    
    # Alternative: Use the current bin occupancies to estimate the "typical" item size?
    # No, bins are sums.
    
    # Let's stick to the current family: Threshold-based.
    # Current: item > 50 -> Worst Fit, else Best Fit.
    # Maybe the threshold should be dynamic?
    # Or maybe we should use a smooth blend?
    
    # Let's try a smooth blend of Best-Fit and Worst-Fit.
    # Score = w * (-remaining) + (1-w) * (remaining)
    # If w=1, Best Fit. If w=0, Worst Fit.
    # We want Best Fit for small items, Worst Fit for large items.
    # So w should decrease as item size increases.
    # Let's use a sigmoid or linear ramp.
    
    # item is in [0, 100].
    # Let's say for item < 20, w=1 (Best Fit).
    # For item > 80, w=0 (Worst Fit).
    # In between, linear.
    
    # But wait, if item is very small, Best Fit is good.
    # If item is very large, Worst Fit is good (to leave space for other large items? No, large items need space, so we want to put them in bins with the MOST space to ensure they fit? No, if they fit, we want to pack them efficiently. But large items fragment space. Putting a large item in a bin with just enough space (Best Fit) might waste the rest if it's too small for anything else. Putting it in a bin with lots of space (Worst Fit) leaves a large chunk that can fit other large items. So Worst Fit for large items makes sense.)
    
    # Let's implement the smooth blend.
    
    # Define the blend factor w
    # w = 1.0 for small items, 0.0 for large items.
    # Let's use a range [20, 80] for the transition.
    low = 20.0
    high = 80.0
    
    if item <= low:
        w = 1.0
    elif item >= high:
        w = 0.0
    else:
        w = (high - item) / (high - low)
    
    # Score = w * (-remaining) + (1-w) * (remaining)
    #      = w * (-remaining) + (1-w) * remaining
    #      = remaining * (1-w - w)
    #      = remaining * (1 - 2w)
    
    # If w=1 (small item), score = remaining * (-1) = -remaining. (Best Fit: min remaining)
    # If w=0 (large item), score = remaining * (1) = remaining. (Worst Fit: max remaining)
    
    # This is a smooth transition.
    
    scores[valid_idx] = remaining[valid_idx] * (1.0 - 2.0 * w)
    
    return scores
