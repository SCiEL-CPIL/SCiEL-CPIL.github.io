"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: First-fit decreasing approximation with a twist.
    We want to minimize waste. A good heuristic is to prefer bins that
    are already partially filled (to consolidate items) but not too full
    (to avoid blocking future large items).
    
    We use a score that balances:
    1. How full the bin is (higher is better for consolidation)
    2. How much space is left (we want to leave space for future items,
       but not too much, as that indicates poor utilization)
    
    A common effective strategy for online bin packing is "Best Fit Decreasing",
    but since we don't know future items, we use a static heuristic.
    
    Let's try: score = (bins - item) * (bins / max(bins + eps))
    This rewards bins that are fuller but still have enough space.
    
    Actually, a simpler and often strong approach is to minimize the remaining
    space after placing the item, but penalize bins that are almost empty.
    
    Let's use: score = -abs((bins - item) - target)
    where target is some fraction of capacity. But we don't know capacity directly.
    
    Alternative: Use a quadratic penalty for remaining space to encourage filling bins.
    score = -(bins - item)**2
    
    But this might be too aggressive. Let's try a combination.
    
    Actually, the classic "Best Fit" is: choose bin with minimum remaining space.
    That is: score = -(bins - item)
    
    Let's try a slightly different approach: "Worst Fit" is: choose bin with maximum remaining space.
    That is: score = bins - item
    
    Research suggests that for random item sizes, Best Fit is often good.
    But let's try a more nuanced score.
    
    We'll use: score = (bins - item) * (bins / (bins + item + eps))
    This gives higher score to bins that are fuller relative to the item.
    
    Actually, let's stick with a well-known effective heuristic: 
    score = - (bins - item) ** 2  # Minimizes the square of remaining space
    This encourages filling bins more aggressively.
    
    But we need to be careful: bins with not enough capacity should be excluded.
    The problem states: "scores each bin with enough remaining capacity"
    So we only score bins where bins >= item.
    
    For bins with not enough capacity, we return -inf or a very low score.
    """
    remaining = bins - item
    
    # Create a score array, initialize to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # For feasible bins (remaining >= 0), compute score
    feasible = remaining >= 0
    if np.any(feasible):
        # Pure Best Fit: Minimize the remaining space.
        # This is the standard strong baseline for online bin packing with random items.
        # It consolidates items, reducing the number of bins used.
        scores[feasible] = -remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
