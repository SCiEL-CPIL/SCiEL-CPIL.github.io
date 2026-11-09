"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track items seen so far (for median estimation)
_items_seen = []
_start_time = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score means the bin is preferred.
    The item goes to the highest-scoring bin.
    
    Strategy: Next-Item Fit proxy. Score bins by how well their remaining capacity
    after placing the current item matches the median item size seen so far.
    This minimizes fragmentation by preferring bins that will be well-utilized by
    typical future items.
    """
    global _items_seen, _start_time
    
    # Deadline guard
    if time.time() - _start_time > 140:
        # Fallback: simple worst-fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full_like(remaining, -np.inf, dtype=np.float64)
        if np.any(valid):
            scores[valid] = remaining[valid]
        return scores
    
    # Track items seen
    _items_seen.append(item)
    
    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Only bins with enough remaining capacity are valid
    valid = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    if not np.any(valid):
        return scores
    
    # Estimate median item size from items seen so far
    # Use a running estimate to avoid sorting the entire list each time
    if len(_items_seen) <= 100:
        median_size = float(np.median(_items_seen))
    else:
        # Use a sample for efficiency
        sample = _items_seen[-100:]
        median_size = float(np.median(sample))
    
    # For valid bins, score based on how well remaining capacity matches median item size
    # We want remaining capacity to be close to median_size (not too much, not too little)
    # Score = negative squared difference, so higher score = closer match
    # But we also need to ensure remaining >= 0 (already handled by valid mask)
    
    # The ideal remaining capacity is around median_size, so that typical future
    # items can fit without leaving too much waste or causing fragmentation.
    # Score = -(remaining - median_size)^2
    # This gives highest score when remaining ≈ median_size
    
    # However, we should also consider that having SOME extra space is good for
    # robustness. Let's use a asymmetric penalty: penalize large excess more than
    # small deficit (since deficit means the bin is nearly full and may not fit
    # the next typical item).
    
    # Actually, let's use a simpler approach: score = -|remaining - median_size|
    # This is L1 distance, which is robust to outliers.
    
    # But wait - we want to prefer bins with remaining capacity that can fit
    # typical items. If remaining < median_size, it might not fit the next item.
    # So we should penalize remaining < median_size more heavily.
    
    # Let's use: score = remaining - alpha * (remaining - median_size)^2
    # where alpha controls how strongly we prefer matching the median.
    
    # Actually, the directive says: "negative squared difference between their
    # remaining capacity and the median item size"
    # So: score = -(remaining - median_size)^2
    
    diff = remaining[valid] - median_size
    scores[valid] = -(diff ** 2)
    
    return scores
# EVOLVE-BLOCK-END
