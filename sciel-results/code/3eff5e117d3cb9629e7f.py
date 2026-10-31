import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for an arriving item using a hybrid heuristic:
    - Invalid bins (insufficient capacity) get -inf.
    - Valid bins are scored to prefer bins that are "almost full" (small remaining),
      but with a penalty if the remaining capacity is too small relative to the item,
      to avoid "wasting" a bin on a tiny leftover.
    
    The score is designed to maximize the chance of packing efficiently by:
    1. Preferring bins with the least remaining capacity (Best-Fit-like).
    2. Penalizing bins where the remaining capacity is very close to the item size
       (to avoid creating bins with very little slack, which might be bad if future items are large).
    
    Actually, a simpler and more robust approach is to use a weighted combination of:
    - -remaining (prefer small remaining)
    - A penalty for bins where remaining is very small (e.g., remaining < item * 0.1)
    
    But let's try a different approach: use the "Worst-Fit" strategy for some cases?
    No, Worst-Fit is generally bad.
    
    Let's try a "Best-Fit Decreasing" variant? We don't have decreasing order.
    
    Alternative: Use a score that is the negative of the remaining capacity,
    but with a small epsilon added to avoid division by zero or numerical issues.
    
    Let's try: score = -remaining + 0.01 * (bins - remaining)  # i.e., -remaining + 0.01 * used
    This slightly prefers bins that are already used, which might help consolidate.
    
    Actually, let's try a purely different heuristic: "Next-Fit" is not possible.
    
    Let's try "First-Fit Decreasing" is not possible.
    
    Let's try a score based on the ratio of remaining to total capacity?
    score = -remaining / bins  # but bins is remaining, so this is -1 for all? No.
    
    Wait, bins is the remaining capacity. So bins is the free space.
    
    Let's try: score = -remaining**0.5  # Prefer small remaining, but with diminishing returns.
    
    Or: score = -1 / (remaining + 0.1)  # Prefer small remaining, but avoid zero.
    
    Let's try: score = -remaining + 0.05 * (100 - remaining)  # Prefer small remaining, but reward used capacity.
    This is equivalent to -0.95 * remaining + 5.
    So it's still Best-Fit, just scaled.
    
    Let's try a completely different approach: "Randomized Best-Fit" with a bias towards bins that are "medium" filled?
    
    Actually, let's try "Worst-Fit" for large items and "Best-Fit" for small items?
    If item > 50, use Worst-Fit (prefer most remaining).
    If item <= 50, use Best-Fit (prefer least remaining).
    
    This is a hybrid heuristic that might work well for Weibull distributions.
    
    Implementation:
    - Compute remaining = bins - item
    - valid = remaining >= 0
    - If item > 50: score = remaining (prefer most remaining)
    - Else: score = -remaining (prefer least remaining)
    - Invalid bins get -inf.
    
    Let's implement this.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    if item > 50:
        # For large items, prefer bins with the most remaining capacity (Worst-Fit)
        scores = np.where(valid, remaining, -np.inf)
    else:
        # For small items, prefer bins with the least remaining capacity (Best-Fit)
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
