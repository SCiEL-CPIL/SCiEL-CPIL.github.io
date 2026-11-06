"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity.
    
    Uses a best-fit strategy combined with a fragmentation penalty based on
    the harmonic mean of remaining capacity and item size. This penalizes both
    tiny leftover gaps (waste) and excessively large remaining spaces
    (risk of future fragmentation), approximating a best-fit-decreasing ideal.
    """
    remaining = bins - item
    mask = remaining >= 0
    
    scores = np.full(bins.shape, -np.inf)
    if not np.any(mask):
        return scores
    
    r = remaining[mask]
    
    # Primary signal: best-fit (minimize remaining capacity)
    # We want higher score = better, so we use -r as base
    
    # Fragmentation penalty via harmonic mean of r and item
    # harmonic_mean = 2 * r * item / (r + item)
    # When r << item: harmonic_mean ≈ 2*r (small, good fit but tiny gap = waste)
    # When r >> item: harmonic_mean ≈ 2*item (large gap = fragmentation risk)
    # We want to penalize both extremes relative to the "ideal" ratio
    
    # Ideal: r should be proportional to item. 
    # Use the ratio r/item and penalize deviation from an optimal range.
    # The harmonic mean approach: score the "efficiency" of the fit.
    
    # Compute a fragmentation score:
    # Let x = r / item (normalized remaining)
    # We want x to be in a "sweet spot" - not too small (waste), not too large (fragmentation)
    # Use: -log(x + 1) as a smooth penalty that grows for both extremes
    # Actually, let's use the harmonic mean directly as a proxy for "fit quality"
    
    # Harmonic mean of r and item:
    # H = 2*r*item / (r + item)
    # This is maximized when r ≈ item, and decreases when r << item or r >> item
    # We want to MAXIMIZE H (good fit) and MINIMIZE r (best-fit)
    
    # Combined score: primary = -r (best-fit), secondary = +H (prefer proportional fit)
    # But we need to balance the scales. Let's normalize.
    
    # Alternative: use the ratio directly
    ratio = r / (item + 1e-12)
    
    # Penalty for fragmentation:
    # If ratio is very small (< 0.1): tiny gap, likely wasted space
    # If ratio is very large (> 5.0): large gap, risk of future fragmentation
    # Use a smooth penalty: -log(1 + |ratio - 1|) or similar
    
    # Let's use: score = -r - alpha * fragmentation_penalty
    # where fragmentation_penalty penalizes deviation from ratio ≈ 1
    
    # Smooth penalty using log: penalizes both small and large ratios
    # penalty = log(1 + (ratio - 1)^2) - this is symmetric around ratio=1
    # But we might want asymmetric: small ratios are worse (waste) than large ones
    
    # Use: penalty = log(1 + ratio) for large ratios (fragmentation)
    #       penalty = log(1 + 1/ratio) for small ratios (waste) 
    # Combined: penalty = max(log(1+ratio), log(1+1/ratio))
    # But max is nonsmooth. Use soft-max: log(exp(a) + exp(b)) - log(2)
    
    a = np.log1p(ratio)           # penalizes large ratio (fragmentation)
    b = np.log1p(1.0 / (ratio + 1e-12))  # penalizes small ratio (waste)
    
    # Soft-max to smoothly combine both penalties
    # This is high when either a or b is high, i.e., when ratio is far from 1
    soft_penalty = np.logaddexp(a, b) - np.log(2.0)
    
    # Scale the penalty to be comparable to r
    # r is in [0, capacity-item], typically O(item)
    # soft_penalty is O(log(ratio)) which is smaller
    # Use a weight to balance
    weight = item * 0.1  # scale penalty to be comparable to remaining
    
    scores[mask] = -r - weight * soft_penalty
    
    return scores
# EVOLVE-BLOCK-END
