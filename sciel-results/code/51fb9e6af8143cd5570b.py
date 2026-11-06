import numpy as np

# State for the running statistics
_item_count = 0
_item_sum = 0.0
_item_sq_sum = 0.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _item_count, _item_sum, _item_sq_sum
    
    # Update statistics
    _item_count += 1
    _item_sum += item
    _item_sq_sum += item * item
    
    # Calculate mean and standard deviation
    mean_item = _item_sum / _item_count
    if _item_count > 1:
        var = max(0, _item_sq_sum / _item_count - mean_item * mean_item)
        std_item = np.sqrt(var)
    else:
        std_item = mean_item  # If only one item, assume std is mean (conservative)
        
    # Identify valid bins (enough capacity)
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    # For valid bins, calculate the remaining space after placing the item
    next_remaining = remaining[valid_mask]
    
    # Strategy: "Next-Item Awareness"
    # We want to choose a bin such that the remaining space (next_remaining)
    # is likely to be used by the next item.
    # The next item is modeled as a normal distribution N(mean_item, std_item).
    # We want to maximize the probability that the next item fits in next_remaining.
    # P(X <= next_remaining) where X ~ N(mean_item, std_item).
    # This is the CDF of the normal distribution.
    # To avoid scipy dependency issues or slowness, we can approximate or use a simpler heuristic.
    # However, scipy is available. Let's use scipy.stats.norm.cdf if available, 
    # but for speed, a simple heuristic might be better.
    
    # Alternative structural idea:
    # If we leave exactly mean_item space, we maximize the chance of fitting the next item.
    # But if we leave too little, we might waste the bin if the next item is larger.
    # If we leave too much, we might be able to fit multiple items, which is good.
    
    # Let's combine Best Fit (minimize remaining) with a penalty for "dead space".
    # Dead space is remaining space that is too small to fit a typical item.
    # If next_remaining < mean_item / 2, it's likely dead space.
    
    # Let's try a scoring function that:
    # 1. Prefers bins where next_remaining is close to mean_item (to fit next item).
    # 2. Strongly penalizes bins where next_remaining is very small (fragmentation).
    # 3. Slightly prefers bins where next_remaining is large (flexibility), but less than best fit.
    
    # Actually, the current program tries to match next_remaining to mean_item.
    # Let's refine this.
    # Score = -|next_remaining - mean_item|
    # This is what the current program does.
    
    # Let's try a different structural change:
    # Use the concept of "Waste".
    # Total waste = sum of remaining capacities.
    # We want to minimize total waste.
    # But we also want to avoid creating "micro-waste" (small gaps).
    
    # Let's try a quadratic penalty for small gaps.
    # If next_remaining < threshold, apply a heavy penalty.
    # Threshold = mean_item * 0.5
    
    threshold = mean_item * 0.5
    # Penalty for being below threshold: (threshold - next_remaining)^2
    # Score = -next_remaining - penalty
    
    # Let's try:
    # Score = -next_remaining - alpha * max(0, threshold - next_remaining)^2
    
    # But we need to tune alpha.
    # Let's try a simpler approach:
    # Score = -next_remaining / (next_remaining + epsilon)
    # This is a monotonic decreasing function of next_remaining.
    # It saturates at -1 for large next_remaining.
    # It approaches 0 for small next_remaining.
    # This favors small next_remaining, but not as aggressively as -next_remaining.
    
    # Let's stick to the "Next-Item Awareness" directive.
    # The probability that a next item (size ~ mean) fits in next_remaining.
    # If we assume next item is uniform [0, 2*mean], then P(fit) = min(1, next_remaining / (2*mean)).
    # Score = P(fit) - next_remaining (to also favor small remaining).
    
    # Let's try:
    # Score = -next_remaining + beta * min(1, next_remaining / (2*mean_item))
    
    # If next_remaining is small, the second term is small, so score is dominated by -next_remaining (good).
    # If next_remaining is large, the second term is beta, so score is -next_remaining + beta.
    # This reduces the penalty for large remaining capacities if they are large enough to fit a next item.
    
    beta = 0.5 * mean_item
    
    fit_prob = np.minimum(1.0, next_remaining / (2.0 * mean_item))
    scores[valid_mask] = -next_remaining + beta * fit_prob
    
    return scores
