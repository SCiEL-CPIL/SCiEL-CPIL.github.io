"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track global time budget
_start_time = None
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing the given item.
    Higher score = better bin.
    
    Strategy: Hybrid heuristic that balances:
    1. Best-fit: minimize leftover space to reduce fragmentation
    2. Avoid tiny fragments: if leftover is too small, it's wasted
    3. Proportionality: favor bins where item size is proportional to remaining space
    
    We use a scoring function that combines:
    - Negative of remaining space after placement (best-fit component)
    - Penalty for very small leftovers (fragmentation avoidance)
    - Bonus for moderate remaining space (flexibility for future items)
    """
    elapsed = _get_elapsed()
    if elapsed > _TIME_LIMIT - 5:
        # Fallback to simple best-fit if close to timeout
        remaining = 1.0 - bins
        valid_mask = remaining >= item
        scores = np.full_like(remaining, -np.inf)
        scores[valid_mask] = -(remaining[valid_mask] - item)
        return scores

    # Calculate remaining space in each bin
    remaining = 1.0 - bins
    
    # Filter out bins that cannot fit the item
    valid_mask = remaining >= item
    
    # Initialize scores to -infinity for invalid bins
    scores = np.full_like(remaining, -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, compute post-placement remaining space
    post_placement = remaining[valid_mask] - item
    
    # Component 1: Best-fit - prefer smaller post-placement space
    # But avoid going to zero (tiny fragments)
    # Use a smooth penalty: -post_placement + epsilon penalty for very small values
    
    # Component 2: Fragmentation penalty - penalize when post_placement is very small
    # A gap < ~0.1 of capacity is likely useless for future items
    frag_threshold = 0.08
    frag_penalty = np.where(post_placement < frag_threshold, 
                            -(frag_threshold - post_placement) * 2.0, 
                            0.0)
    
    # Component 3: Flexibility bonus - moderate remaining space is good
    # We want to avoid both tiny gaps and leaving too much space
    # Ideal post-placement is around 0.1-0.3 (heuristic)
    # Use a Gaussian-like bonus centered at 0.15
    center = 0.15
    width = 0.12
    flex_bonus = np.exp(-((post_placement - center) ** 2) / (2 * width ** 2))
    
    # Combine: primary driver is best-fit (minimize leftover), with adjustments
    # Score = -post_placement (best fit) + frag_penalty + alpha * flex_bonus
    # The -post_placement dominates; frag_penalty and flex_bonus are secondary
    alpha = 0.3
    
    valid_scores = -post_placement + frag_penalty + alpha * flex_bonus
    
    scores[valid_mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
