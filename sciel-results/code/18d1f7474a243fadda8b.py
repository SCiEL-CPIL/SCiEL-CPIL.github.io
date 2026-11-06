"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Structural change: Use a bimodal scoring function that prefers bins that
    become either nearly full (sealed) or retain substantial remaining capacity
    (open for large future items). This avoids the "middle" fragmentation zone
    where remaining space is too small for most items but wastes a bin slot.
    
    For Weibull-distributed items (skewed, many small, few large), the key
    insight is that remaining space below ~15-20 units is often useless.
    So we want to either:
    1. Fill bins to near-completion (remaining < ~5): good, bin is sealed
    2. Keep bins with large remaining (> ~30): good, can fit large items
    3. Avoid bins that would leave 5-30 remaining: bad, fragmented
    
    Score = -fragmentation_cost(remaining)
    where fragmentation_cost is low for very small and very large remaining,
    and high for medium remaining.
    """
    t0 = time.time()
    
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        rem = remaining[valid]
        
        # Bimodal fragmentation cost:
        # Low cost when rem is very small (bin nearly sealed) or very large (bin stays open)
        # High cost when rem is in the "dead zone" (10-40)
        
        # Use a smooth bimodal penalty:
        # cost = exp(-(rem - 25)^2 / (2 * 15^2))  -- Gaussian bump centered at 25
        # This penalizes remaining around 25, rewards rem near 0 or rem > 50
        
        center = 25.0
        width = 15.0
        gaussian_penalty = np.exp(-(rem - center) ** 2 / (2.0 * width ** 2))
        
        # Also add a small base penalty proportional to remaining to prefer sealing
        base_penalty = rem * 0.01
        
        # Total score: negative of total cost
        scores[valid] = -(base_penalty + 10.0 * gaussian_penalty)
    
    # Deadline guard
    if time.time() - t0 > 140:
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
