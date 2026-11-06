"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Uses a harmonic-mean-inspired fragmentation penalty:
    - Penalizes bins that are too large (future fragmentation risk)
    - Penalizes bins that leave tiny gaps (waste)
    - Favors bins where remaining capacity after placement is 'healthy'
    
    The score is negative of a weighted combination that encourages
    bins where the item fills a substantial but not excessive portion.
    """
    remaining = bins - item
    # Only bins with enough capacity are valid
    valid = remaining >= 0
    
    # For valid bins, compute a fragmentation-aware score
    # We want to minimize fragmentation: avoid both very small and very large remainders
    # after placing the item.
    
    # Use a smooth surrogate: for valid bins, score = -1/sqrt(remaining + eps) 
    # which penalizes small remainders (high score = preferred) but also
    # we want to avoid too-large remainders.
    
    # Better approach: penalize deviation from an ideal remaining capacity.
    # Ideal: remaining should be small enough to not waste space, but not so small
    # that it creates unusable fragments.
    
    # Strategy: score = -(remaining + item * 0.1) / (remaining + item + 1)
    # This gives higher scores (less negative) to bins with smaller remaining capacity,
    # but with diminishing returns, avoiding the extreme preference for near-zero gaps.
    
    # Alternative: use the ratio of item to bin capacity as a fit metric
    # fit = item / bins (for valid bins)
    # We want fit to be high (item fills the bin well) but not necessarily 1.0
    
    # Let's use: score = -remaining * (1 - item/bins) for valid bins
    # This penalizes large remainders more when the item is small relative to the bin
    
    # Simpler and effective: negative of a smooth function that peaks at moderate remaining
    # Use: score = -1 / (remaining + 1) + 0.5 * (item / (bins + 1))
    # The first term prefers smaller remaining, the second prefers larger item-to-bin ratio
    
    # Let me try a cleaner approach inspired by the harmonic mean idea:
    # harmonic_mean(remaining, item) = 2 * remaining * item / (remaining + item)
    # We want this to be large (good fit), so score = -1/harmonic_mean for valid bins
    # But harmonic_mean is 0 when either is 0, so 1/harmonic_mean is inf -> use safe version
    
    # Actually, let's use: for valid bins, score = -(remaining + item) / (2 * sqrt(remaining * item + 1))
    # This is related to the arithmetic-geometric mean and penalizes imbalance
    
    # Let me go with a practical approach:
    # score = -remaining / (remaining + 10) for valid bins
    # This gives a smooth, bounded penalty that prefers smaller remainders
    
    # Even better: combine best-fit with a small bonus for bins that are not too large
    # score = -remaining + 0.1 * item  (but this is just best-fit with offset)
    
    # Final approach: use a smooth best-fit with fragmentation awareness
    # For valid bins: score = -remaining * (1 + 0.5 * (1 - item / (bins + 1e-9)))
    # This amplifies the penalty for large remainders when the item is small relative to bin
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        rem = remaining[valid]
        bin_vals = bins[valid]
        item_safe = max(item, 1e-9)
        
        # Fit ratio: how well the item fits in the bin
        fit_ratio = item_safe / (bin_vals + 1e-9)
        
        # Base score: prefer smaller remaining (best-fit)
        base_score = -rem
        
        # Fragmentation penalty: penalize bins that are much larger than needed
        # If fit_ratio is small, the bin is much larger than the item -> penalty
        frag_penalty = 0.5 * rem * (1 - fit_ratio)
        
        # Combined score
        scores[valid] = base_score - frag_penalty
    
    return scores
# EVOLVE-BLOCK-END
