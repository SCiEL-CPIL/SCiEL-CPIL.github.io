"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track bin usage statistics across calls
_bin_fill_history = None  # Will be initialized on first call
_capacity = 100.0
_start_time = None


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses First-Fit-Diminishing heuristic:
    Score = -remaining capacity (pure best-fit)
    This aggressively consolidates items into bins with least remaining space,
    keeping other bins empty for larger future arrivals.
    """
    global _start_time, _capacity
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're getting close to the 150s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        # Emergency: just return best-fit scores quickly
        remaining = bins - item
        valid_mask = remaining >= 0
        scores = np.full_like(remaining, -np.inf, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores
    
    # Track remaining capacity for each bin
    remaining = bins - item
    
    # Filter to only bins that can hold the item
    valid_mask = remaining >= 0
    
    # Create score array with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    # If only one valid bin, return a neutral score
    if n_valid == 1:
        scores[valid_mask] = 0.0
        return scores
    
    # Best-fit with refined penalty structure
    # Primary score: negative remaining capacity (tightest fit first)
    valid_scores = -valid_remaining
    
    # Refined secondary penalty:
    # 1. Strongly prefer bins where the item fills >50% of remaining space
    # 2. Penalize bins where item is very small relative to remaining space
    # 3. Slight preference for bins that are already partially filled (consolidation)
    
    capacity = _CAPACITY
    
    # Normalize remaining to [0, 1]
    norm_remaining = valid_remaining / capacity
    
    # Penalty 1: Encourage filling bins that are close to full
    # When remaining is small (bin nearly full), give bonus
    fill_bonus = 0.1 * (1.0 - norm_remaining)
    
    # Penalty 2: Penalize leaving large gaps
    # When item is much smaller than remaining, penalize
    gap_penalty = 0.05 * np.maximum(0, norm_remaining - 0.5)
    
    # Penalty 3: Small bonus for bins that have been used (have some fill)
    # This encourages consolidation into already-open bins
    # We approximate "used" by checking if remaining < capacity (i.e., bin has items)
    # Since we don't track history, use a smooth approximation
    usage_bonus = 0.02 * (1.0 - norm_remaining) ** 2
    
    # Combine: primary (tight fit) + secondary adjustments
    valid_scores += fill_bonus - gap_penalty + usage_bonus
    
    # Add a tiny deterministic tie-breaker to ensure reproducibility
    # The tie-breaker is so small it doesn't affect the primary decision
    tie_breaker = np.linspace(0, 1e-9, n_valid)
    valid_scores += tie_breaker
    
    scores[valid_mask] = valid_scores
    
    return scores


# Pre-compute some parameters that might be useful
# The capacity is given as 100 in the task description
_CAPACITY = 100.0

# EVOLVE-BLOCK-END
