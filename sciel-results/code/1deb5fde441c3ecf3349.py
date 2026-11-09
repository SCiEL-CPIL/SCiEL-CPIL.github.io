"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    Returns a score array where higher is better.
    Only bins with enough remaining capacity get a finite score; others get -inf.
    
    Strategy: Smooth best-fit with a quadratic penalty on wasted space,
    combined with a small consolidation bonus for already-used bins.
    Uses a smooth exponential decay scaled by bin capacity to handle
    different fill levels gracefully.
    """
    # Deadline guard: track elapsed time for safety
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=np.float64)
    
    # Compute remaining capacity after placing item
    remaining = bins - item
    
    # Only consider bins where remaining >= 0 (enough capacity)
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Core scoring: smooth best-fit using exponential decay
    # Higher score = tighter fit (less remaining space)
    # Scale parameter tau relative to bin capacity for consistent behavior
    # tau = 10.0 works well for capacity 100 bins
    tau = 10.0
    
    # Primary score: exponential decay of remaining space
    # remaining = 0 -> score = 1.0 (perfect fit)
    # remaining = tau -> score = exp(-1) ≈ 0.37
    # remaining = 2*tau -> score = exp(-2) ≈ 0.14
    base_score = np.exp(-valid_remaining / tau)
    
    # Secondary score: quadratic penalty on wasted space
    # This provides a stronger distinction between close fits
    # Normalized by capacity for scale invariance
    capacity = 100.0
    normalized_remaining = valid_remaining / capacity
    quad_penalty = -(normalized_remaining ** 2)
    
    # Tertiary score: consolidation bonus
    # Prefer bins that are already partially used (remaining < capacity)
    # This helps reduce the number of open bins
    # Bonus is small to not override the primary best-fit signal
    fill_fraction = 1.0 - (valid_remaining / capacity)
    consolidation_bonus = 0.05 * fill_fraction
    
    # Combine scores with weighted sum
    # Primary (exponential) dominates, quadratic refines, consolidation is minor
    combined = 1.0 * base_score + 0.3 * (quad_penalty + 1.0) + consolidation_bonus
    
    # Alternative: use a multiplicative combination for smoother interaction
    # This avoids the additive cancellation that can occur with negative terms
    # multiplicative_score = base_score * (1.0 + 0.2 * fill_fraction)
    # multiplicative_score *= (1.0 - 0.1 * normalized_remaining)
    
    # Use the additive combination as it's more interpretable and stable
    scores[valid_mask] = combined
    
    # Check deadline - if we're close to the limit, return immediately
    # (The computation above is O(n) and very fast, so this is just a safety net)
    if time.time() > DEADLINE:
        return scores
    
    # Optional: apply a final polish pass with a different scale for tie-breaking
    # This helps distinguish between very close fits
    # Find bins with very similar scores and break ties using a finer-grained metric
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) > 1:
        score_range = np.max(finite_scores) - np.min(finite_scores)
        if score_range < 1e-6:
            # Scores are nearly tied; use a finer-grained best-fit
            # with a smaller tau for better discrimination
            tau_fine = 2.0
            fine_scores = np.exp(-valid_remaining / tau_fine)
            scores[valid_mask] = fine_scores
    
    return scores
# EVOLVE-BLOCK-END
