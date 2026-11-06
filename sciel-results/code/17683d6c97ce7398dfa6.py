"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching and deadline management
_start_time = time.time()
_DEADLINE = 150.0  # Wall-clock limit
_SAFETY_MARGIN = 1.0  # Seconds before deadline to stop
_cache = {}


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Returns an array of scores where higher is better.
    The item goes to the highest-scoring bin.
    
    Strategy: Next-Item Awareness heuristic
    - Estimate the expected size of the next item from the current bin occupancy distribution
    - Select the bin that minimizes the probability of wasting capacity
    - Score bins based on how well their remaining space matches this predicted next item size
    """
    global _start_time, _cache
    
    # Deadline guard: if we're close to the limit, use a fast fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fast fallback: best-fit
        remaining = bins - item
        scores = np.full_like(remaining, -1e9, dtype=np.float64)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores
    
    # Check if we have a cached result for this state
    # We use a hash of the bins array and item
    try:
        # Only use cache for small arrays to avoid memory issues
        if len(bins) < 1000:
            key = (item, tuple(bins.tolist()))
            if key in _cache:
                return _cache[key]
    except:
        pass
    
    remaining = bins - item
    scores = np.full_like(remaining, -1e9, dtype=np.float64)
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    # Get the remaining capacities of valid bins
    valid_remaining = remaining[valid]
    
    # Estimate the expected size of the next item
    # Heuristic: The average item size can be estimated from the current usage pattern
    # If bins are being filled, the average item size is roughly the average filled amount
    # But we don't have access to the full history, so we use a simpler heuristic
    
    # Heuristic 1: Use the median of the remaining capacities as a proxy for typical item sizes
    # If most bins have similar remaining capacity, items are likely similar in size
    # The next item is likely to be similar to the current item or the median of what's been placed
    
    # A better heuristic: The expected next item size is related to the distribution of
    # already-placed items. Since we can't see history, we approximate:
    # - If the current item is small, the next item might also be small
    # - If the current item is large, the next item might be smaller (Weibull distribution)
    
    # For Weibull-distributed items with capacity 100, typical items are in the range 1-50
    # The mode of a Weibull distribution depends on shape and scale parameters
    
    # Simple approach: Assume the next item has a similar size to the current item
    # but with some smoothing. We score bins by how well their remaining capacity
    # can accommodate a range of possible next item sizes.
    
    # Score based on:
    # 1. Best-fit for the current item (prefer tighter fit)
    # 2. Leave enough room for a likely next item
    # 3. Avoid leaving tiny gaps that can't be used
    
    # Estimated next item size: use a weighted combination of current item and a baseline
    # For Weibull distributions, the mean is typically larger than the mode
    # Let's estimate the next item as 0.8 * current_item + 2.0 (small offset for variation)
    
    # Actually, a better approach: Score based on the "waste" after placing the current item
    # and a hypothetical next item. We want to minimize the probability that the remaining
    # space after the current item is too small for the next item.
    
    # Let's use a smooth approximation:
    # For each valid bin, compute:
    #   waste = remaining_capacity_after_current - estimated_next_item
    # If waste is very negative, the next item won't fit (bad)
    # If waste is very positive, we're wasting space (also bad)
    # Optimal waste is close to 0
    
    # Estimate next item size:
    # Use the current item as a baseline, but adjust based on the distribution
    # For Weibull, if current item is small, next item might be larger
    # If current item is large, next item might be smaller
    
    # Simple heuristic: next_item_estimate = max(1.0, 0.7 * item + 3.0)
    next_item_est = max(1.0, 0.7 * item + 3.0)
    
    # For each valid bin, compute the remaining space after placing the current item
    # Then score based on how well this remaining space matches the estimated next item
    
    # Score components:
    # 1. Tightness of fit for current item: prefer smaller remaining (best-fit)
    # 2. Match with next item: prefer remaining space close to next_item_est
    # 3. Penalty for very small gaps (can't fit even small items)
    
    tightness = -valid_remaining  # Best-fit: prefer smaller remaining
    
    # Match score: how well the remaining space matches the estimated next item
    # Use a Gaussian-like score: exp(-((remaining - next_item_est)^2) / (2 * sigma^2))
    sigma = 5.0  # Standard deviation for the match score
    match_score = np.exp(-((valid_remaining - next_item_est) ** 2) / (2 * sigma ** 2))
    
    # Penalty for very small gaps (less than 2.0)
    small_gap_penalty = np.where(valid_remaining < 2.0, -5.0, 0.0)
    
    # Combine scores with weights
    weight_tightness = 1.0
    weight_match = 2.0
    weight_penalty = 1.0
    
    combined = (weight_tightness * tightness + 
                weight_match * match_score * 10.0 +  # Scale match score
                weight_penalty * small_gap_penalty)
    
    scores[valid] = combined
    
    # Cache the result
    try:
        if len(bins) < 1000:
            key = (item, tuple(bins.tolist()))
            _cache[key] = scores.copy()
    except:
        pass
    
    return scores
# EVOLVE-BLOCK-END
