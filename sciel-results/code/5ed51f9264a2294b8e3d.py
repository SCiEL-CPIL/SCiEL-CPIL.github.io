"""Fragmentation-aware online bin packing heuristic."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing `item`. Higher score = better placement.
    
    Uses a fragmentation-aware model:
    - We want to fill bins to reduce waste, but avoid creating tiny fragments.
    - The L1 lower bound suggests we want to pack as tightly as possible.
    - A good heuristic is to prefer bins where the remaining space after placement
      is either very small (bin nearly full) or large enough to accommodate future items.
    
    Strategy: Score = -(remaining capacity after placement) + bonus for avoiding fragmentation
    
    Specifically, we penalize bins where the item takes a moderate fraction of remaining space,
    leaving a small unusable fragment. We reward bins where either:
    1. The bin becomes nearly full (little waste)
    2. The remaining space is still substantial (can fit future items)
    
    We use a smooth function that peaks when remaining space is either ~0 or large.
    """
    # Remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    # For invalid bins, assign a very low score
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
    
    rem_valid = remaining[valid]
    
    # Item size relative to remaining capacity before placement
    # ratio = item / (rem_valid + item) = item / bins[valid]
    # But we can also think in terms of how full the bin will be after placement
    
    # Fill ratio after placement: how much of the bin is used
    # fill_after = 1 - rem_valid / bins[valid]
    # But we don't have the original bin capacities directly... 
    # Wait, bins is the array of remaining capacities, not total capacities.
    
    # Actually, re-reading the problem: "bins" is an np.ndarray where each element 
    # represents the remaining capacity of that bin. So bins[i] is how much space is left.
    
    # So after placing item, remaining space is rem_valid.
    # We want to minimize waste, but also avoid creating tiny fragments.
    
    # A simple approach: score = -rem_valid (best-fit)
    # Better: penalize small remaining spaces that are too small to be useful
    
    # Let's define a "fragment penalty" that is high when rem_valid is in a "bad" range
    # Bad range: small but not tiny (e.g., 1-10 units when capacity is 100)
    
    # Use a Gaussian-like penalty centered around a "bad" fragment size
    # Or use a function that rewards rem_valid being close to 0 or close to some threshold
    
    # Let's try: score = -rem_valid + alpha * exp(-((rem_valid - mu)^2) / (2*sigma^2))
    # where mu is the "bad" fragment size we want to avoid
    
    # Actually, simpler: use a piecewise approach
    # If rem_valid < threshold, it's good (bin nearly full)
    # If rem_valid > threshold2, it's okay (still useful)
    # If threshold < rem_valid < threshold2, it's bad (fragment)
    
    # Let's use a smooth approximation
    # score = -rem_valid * (1 + beta * exp(-((rem_valid - mu)^2) / (2*sigma^2)))
    
    # With mu ~ 5, sigma ~ 3, beta ~ 1.0
    # This penalizes bins where remaining space is around 5 units
    
    # Waste-aware scoring: minimize total expected waste
    # Strategy: prefer placements that leave either very small or very large remaining space
    # Use a smooth function that has minima at rem=0 and rem=large values
    
    # Approach: score based on how "useful" the remaining space is
    # Small remaining (< threshold): good (bin nearly full)
    # Large remaining (> threshold2): okay (can fit more items)
    # Medium remaining: bad (fragmentation)
    
    # Use a piecewise smooth function:
    # For rem < T1: cost ~ rem (nearly linear, slight penalty for not being exactly 0)
    # For T1 <= rem <= T2: cost ~ rem + penalty (fragmentation zone)
    # For rem > T2: cost ~ rem * alpha (reduced penalty for large remaining)
    
    T1 = 3.0   # below this, it's "good enough"
    T2 = 15.0  # above this, remaining space is still useful
    alpha = 0.7  # reduced weight for large remaining spaces
    
    # Compute a smooth transition
    # Use sigmoid-like functions for smooth transitions
    eps = 0.1
    
    # Transition 1: from linear to penalized zone
    # At rem < T1, weight = 1.0; at rem > T1+eps, weight increases
    trans1 = 0.5 * (1.0 + np.tanh((rem_valid - T1) / eps))
    
    # Transition 2: from penalized to reduced zone  
    # At rem < T2-eps, weight = 1.0; at rem > T2, weight = alpha
    trans2 = 0.5 * (1.0 + np.tanh((rem_valid - T2) / eps))
    
    # Base cost: rem_valid
    # In fragmentation zone (T1 to T2), add extra penalty
    frag_penalty = 2.0 * np.exp(-((rem_valid - (T1+T2)/2) ** 2) / (2 * ((T2-T1)/4) ** 2))
    
    # Combined cost with smooth transitions
    cost = rem_valid * (1.0 + trans1 * frag_penalty) * (1.0 - (1.0-alpha) * trans2)
    
    # Additional: penalize items that take a very small fraction of space
    # (wastes capacity by not filling bins efficiently)
    ratios = item / bins[valid]
    small_ratio_penalty = 0.5 * np.exp(-((ratios - 0.1) ** 2) / (2 * 0.05 ** 2))
    
    cost += 5.0 * small_ratio_penalty
    
    scores[valid] = -cost
    
    return scores
# EVOLVE-BLOCK-END
