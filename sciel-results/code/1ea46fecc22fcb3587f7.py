"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0 - 5.0  # 145s safety margin


def _check_deadline():
    return time.time() > _DEADLINE


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins using a composite heuristic for online bin packing.
    
    The score combines:
    1. Best-fit-decreasing intuition: prefer bins where the item fits tightly
       (minimizes leftover waste in that bin).
    2. A penalty for creating "dead space" that is too small to be useful
       for future items (based on typical Weibull item sizes).
    3. A slight preference for bins that are already partially filled
       (conservation of empty bins for large items).
    
    Higher score = more preferred.
    """
    # Fast path: deadline check
    if _check_deadline():
        # Fallback to simple best-fit
        remaining = bins - item
        scores = np.zeros(len(bins))
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        scores[~valid] = -np.inf
        return scores

    remaining = bins - item
    valid = remaining >= 0

    # If no valid bins, return all -inf
    if not np.any(valid):
        return np.full(len(bins), -np.inf)

    # For valid bins, compute a composite score
    # We want to maximize the score (item goes to highest-scoring bin)
    
    # Component 1: Best-fit (tighter fit is better)
    # Score is higher when remaining is smaller
    best_fit_score = -remaining[valid]
    
    # Component 2: Avoid creating very small leftovers that are "dead space"
    # For Weibull with capacity 100, typical items are small.
    # If remaining is very small (< some threshold), it's wasted capacity.
    # Penalize bins where the leftover is small but not zero.
    # Actually, best-fit already prefers small leftovers.
    # The issue is when leftover is so small it can't fit any future item.
    # Let's use a different angle: prefer bins where the remaining capacity
    # is either very small (tight fit) or large enough to be useful.
    
    # Component 3: Conservation of empty bins
    # Prefer bins that already have some items (i.e., bins with smaller remaining)
    # This is already captured by best_fit_score.
    
    # Let's try a weighted combination:
    # Primary: minimize remaining (best-fit)
    # Secondary: if remaining is below a threshold, add a small bonus
    # to encourage filling bins almost completely.
    
    # Actually, let's try a simpler but effective heuristic:
    # Score = -remaining + small_bonus_if_remaining_is_very_small
    
    # For Weibull(100), items are typically in [0, 100] with most < 50.
    # A leftover of < 5 is likely useless. A leftover of > 50 is very useful.
    # We want to prefer tight fits, but not at the cost of wasting a bin.
    
    # Let's use: score = -remaining * (1 + alpha * (remaining < threshold))
    # No, let's keep it simple and robust.
    
    # New approach: Use the ratio of item to bin capacity used.
    # Prefer bins where the item takes a large fraction of the remaining capacity.
    # This is equivalent to best-fit.
    
    # Let's try adding a term that penalizes bins where the remaining capacity
    # after placement is in the "dead zone" (e.g., 5 < remaining < 20).
    # This is a heuristic based on typical Weibull item sizes.
    
    dead_zone = (remaining[valid] > 5) & (remaining[valid] < 20)
    
    scores_valid = -remaining[valid]
    # Penalize dead zone leftovers
    scores_valid[dead_zone] -= 5.0
    
    # Construct full scores array
    scores = np.full(len(bins), -np.inf)
    scores[valid] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
