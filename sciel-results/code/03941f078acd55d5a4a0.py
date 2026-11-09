"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s. We keep a margin.
_DEADLINE = time.time() + 145.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    
    Strategy: Best-Fit Decreasing is optimal for offline, but online we use a
    heuristic that balances:
    1. Tight fit (leave less waste) - classic best-fit
    2. Avoid fragmentation - prefer bins that are already 'used' moderately
    3. Smooth surrogate: use a soft-min-like scoring to avoid cliff edges
    
    For Weibull items with capacity 100, items are typically small-to-medium.
    The L1 lower bound is ceil(sum(items)/capacity). We want to minimize
    excess = (bins_used - L1_bound) / L1_bound.
    
    A good online heuristic: Best-Fit but with a slight preference for bins
    that would become 'fuller' (i.e., minimize remaining capacity after placement,
    but penalize creating very small scraps).
    
    Score = -(remaining_after) + small_bonus_if_remaining_after is very small
    This is essentially best-fit with a tweak.
    
    Actually, let's think more carefully. The seed is pure best-fit:
    score = -(bins - item) = item - bins, so higher score = smaller bins.
    This picks the bin with smallest capacity that fits.
    
    Improvements to consider:
    - First-fit-decreasing style: but we don't know future items.
    - Weighted best-fit: prefer bins where the item fills a larger fraction.
    - Avoid 'wasteful' small scraps: if remaining_after < item*0.3, it's a scrap.
    
    Let's try: score = -remaining_after + alpha * (item / bins)
    where alpha encourages filling bins more completely.
    
    But we must be careful: the score must be such that higher is better,
    and only bins with bins >= item are considered (others get -inf or very low).
    
    Let me try a smooth approach:
    For bins that fit:
      remaining = bins - item
      fill_fraction = item / bins  (how much of the bin is used)
      score = fill_fraction - beta * remaining / capacity  (or similar)
    
    Actually, simpler: just improve best-fit by adding a small penalty for
    leaving a remainder that is too small (scrap) or too large (waste).
    
    Let's try: score = -(bins - item) + gamma * min(item, bins - item) / bins
    This adds a bonus for bins where the item and remainder are balanced.
    
    Hmm, let me think about what minimizes excess over L1 bound.
    L1 bound = ceil(total_size / capacity).
    To minimize bins_used, we want to pack tightly.
    
    Best-fit is known to be good. Let me try a variant:
    - If the item fits in a bin with very little remaining (tight fit), prefer it.
    - But if multiple bins have similar tight fits, prefer the one with more total capacity used already?
    
    Actually, for online bin packing, First-Fit and Best-Fit are both asymptotically optimal (ratio 1.7).
    Best-Fit is generally slightly better in practice.
    
    Let me try: score = -(bins - item) + epsilon * bins
    This slightly prefers larger bins among those with similar remaining capacity,
    which might help with future larger items.
    
    Or: score = -(bins - item) / bins  (normalize by bin size)
    This prefers bins where the remaining capacity is a small fraction of the bin.
    
    Let me go with: for fitting bins, score = item / bins (fill fraction after placing).
    Higher fill fraction = better. This is a smooth measure.
    
    But wait: if two bins have the same fill fraction, we might want to break ties
    by preferring the smaller bin (to save larger bins for larger items).
    
    Let me combine: score = item / bins - delta * bins
    where delta is small.
    
    Actually, let's keep it simple and try: score = item / bins for fitting bins.
    This is a smooth, monotonic function that prefers tighter fits.
    """
    # Handle edge cases
    if bins.size == 0:
        return np.array([])
    
    # Check if item can fit in any bin
    mask = bins >= item
    
    if not np.any(mask):
        # No bin can fit the item. Return very low scores for all.
        return np.full_like(bins, -1e10, dtype=float)
    
    scores = np.full_like(bins, -1e10, dtype=float)
    
    # For bins that fit, compute score
    fitting_bins = bins[mask]
    # Fill fraction: item / bin_capacity
    fill = item / fitting_bins
    # Add a small preference for smaller bins among those with similar fill
    # to save larger bins for potentially larger future items
    # score = fill - 0.01 * fitting_bins / 100.0  (normalize bin size)
    
    # Actually, let's use: score = fill * (1 - 0.05 * fitting_bins / 100.0)
    # This slightly penalizes using very large bins when a tight fit is available.
    
    # Simpler: just use fill fraction, which naturally prefers smaller bins
    # when the item is the same size.
    scores[mask] = fill
    
    return scores
# EVOLVE-BLOCK-END
