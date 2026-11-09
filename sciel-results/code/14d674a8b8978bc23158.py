# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Combine best-fit (tight packing) with a fill-rate bonus
    to encourage completing partially-filled bins, reducing fragmentation.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    # For valid bins, compute scores
    # Base: negative remaining (best-fit: prefer smallest remaining)
    # Bonus: reward bins that are already well-filled (high utilization before)
    # This encourages finishing bins rather than starting new ones
    
    fill_before = 1.0 - (bins / 100.0)  # fraction of bin already used (capacity=100)
    remaining_frac = remaining / 100.0  # fraction of bin remaining after placement
    
    # Score: prefer small remaining (tight fit) but also prefer well-filled bins
    # Use a combination that penalizes medium remaining (fragmentation)
    # and rewards very small or very large remaining
    
    # Penalty for fragmentation: medium-sized gaps are worst
    # We want to avoid leaving gaps that are too small to be useful but too large to ignore
    # A good proxy: score = -remaining + bonus for high fill
    
    # Let's use: score = -remaining * (1 + 0.5 * fill_before)
    # This makes tight fits in well-filled bins even more attractive
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        # For valid bins:
        rem = remaining[valid]
        fill = fill_before[valid]
        
        # Primary: minimize remaining space (best-fit)
        # Secondary: among similar remaining, prefer bins with higher fill
        # Combine: score = -rem * (1.0 + 0.3 * fill)
        # This slightly boosts preference for well-filled bins with tight fits
        
        scores[valid] = -rem * (1.0 + 0.3 * fill)
    
    return scores
# EVOLVE-BLOCK-END
