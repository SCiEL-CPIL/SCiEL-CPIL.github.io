"""Rewritten online bin packing priority scorer."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # Safety margin before 150s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Item goes to the highest-scoring bin.
    
    Strategy: Hybrid of Best-Fit and Worst-Fit with fragmentation penalty.
    - Best-fit minimizes wasted space in the chosen bin (tight fit).
    - Worst-fit keeps bins open for future large items.
    - We penalize medium-sized residuals more heavily to avoid fragmentation.
    
    Score = combination of:
      1. Tightness: prefer bins where residual = bins[i] - item is small (best-fit)
      2. Openness: prefer bins with more remaining capacity (worst-fit) 
      3. Fragmentation penalty: penalize residuals that are "awkward" medium sizes
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple best-fit
        remaining = bins - item
        valid = remaining >= 0
        if not np.any(valid):
            return np.full(len(bins), -np.inf)
        scores = np.full(len(bins), -np.inf)
        scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0

    if not np.any(valid):
        return np.full(len(bins), -np.inf)

    # Only compute for valid bins
    r = remaining[valid]
    
    # Normalize by item size to make scale-invariant
    # r is the residual capacity after placing item
    
    # Component 1: Best-fit preference (smaller residual = better)
    # We want higher score for smaller residual
    best_fit_score = -r
    
    # Component 2: Worst-fit preference (larger residual = better, keeps bin open)
    # But we don't want pure worst-fit as it wastes space
    # Weighted by how "large" the item is relative to bin capacity
    # For small items, worst-fit is good; for large items, best-fit is good
    item_ratio = item / max(bins.max(), 1e-9)
    # When item is small (< 0.2 of capacity), favor worst-fit more
    # When item is large (> 0.5 of capacity), favor best-fit more
    wf_weight = max(0.0, 0.5 - item_ratio) * 2.0  # 0 to 1
    bf_weight = 1.0 - wf_weight  # complement
    
    worst_fit_score = r  # larger residual = better for worst-fit
    
    # Component 3: Fragmentation penalty
    # Penalize residuals that are in the "medium" range
    # Medium residuals create awkward gaps that can't fit many future items
    # We use a peaked penalty at medium residual sizes
    # Normalize residual by typical bin size
    typical_bin = np.median(bins[valid]) if np.any(valid) else 100.0
    norm_r = r / max(typical_bin, 1e-9)
    
    # Penalty peaks around norm_r = 0.3-0.5 (medium gaps)
    # Use a Gaussian-like penalty centered at 0.4
    frag_penalty = -np.exp(-((norm_r - 0.4) ** 2) / (2 * 0.15 ** 2)) * 0.5
    
    # Combine scores
    # Scale best_fit and worst_fit to comparable ranges
    # best_fit_score is negative, worst_fit_score is positive
    # We want the combined score to be higher for better choices
    combined = bf_weight * best_fit_score + wf_weight * worst_fit_score + frag_penalty
    
    # Construct full score array
    scores = np.full(len(bins), -np.inf)
    scores[valid] = combined
    
    return scores
# EVOLVE-BLOCK-END
