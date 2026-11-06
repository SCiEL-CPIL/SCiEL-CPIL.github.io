"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing. Higher score = preferred bin.
    The item goes to the highest-scoring bin among those with enough capacity.
    
    Strategy: A combination of Best-Fit (minimize remaining space after placement)
    with a penalty for opening new bins, tuned for Weibull-distributed items.
    
    For Weibull items with capacity 100, most items are small.
    We want to fill bins efficiently but also avoid wasting capacity in bins
    that are already partially filled if a better fit exists elsewhere.
    
    The score is:
      - If bin has enough capacity: score = -remaining_after_placement + small bonus for fuller bins
      - If bin doesn't have enough capacity: score = -inf (will not be selected)
    
    We use a smooth approximation to encourage filling bins while avoiding
    the "wasted space" problem of pure best-fit.
    """
    # Track deadline to ensure we finish well within 150s
    # This function is called per-item, so it must be fast
    
    remaining = bins - item
    mask = remaining >= 0
    
    # For bins with enough capacity, we want to minimize remaining space (best-fit)
    # but also consider the "waste" - how much capacity is left unused in the bin
    # after placing this item.
    
    # Score: we prefer bins where placing the item leaves the least remaining space.
    # But we also want to avoid creating many nearly-empty bins.
    # A good heuristic: score = -remaining_after + alpha * (bins_used_fraction)
    # where bins_used_fraction encourages filling bins that are already partially used.
    
    # Actually, for online bin packing with unknown future items, the standard
    # good heuristics are:
    # 1. Best Fit: minimize remaining space after placement
    # 2. Best Fit Decreasing (offline, not applicable)
    # 3. First Fit: first bin that fits
    # 4. Worst Fit: maximize remaining space (spreads items)
    
    # For Weibull distributions with many small items, Best Fit tends to work well.
    # But let's add a small refinement: prefer bins that are already "warmed up"
    # (have some usage) to consolidate items and leave fresh bins for larger items.
    
    # remaining_after = bins - item for bins that fit
    # We want to minimize remaining_after, so score = -remaining_after
    
    # But we can add a tiny bonus for bins that have more remaining capacity
    # (to leave room for future items) - this is actually opposite of best-fit.
    
    # Let's stick with a refined best-fit:
    # score = -remaining_after + epsilon * bins (prefer slightly larger bins for robustness)
    
    # Actually, let's think about what minimizes excess over L1 lower bound.
    # L1 lower bound = ceil(total_size / capacity)
    # We want to minimize the number of bins used, which means filling bins as much as possible.
    
    # Best fit does this well. Let's use it with a small tie-breaker.
    
    # Create scores array
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(mask):
        # For bins that fit, score = -remaining_after (minimize remaining)
        # Add a very small penalty proportional to how full the bin already is
        # to encourage spreading when remaining spaces are equal (avoids clustering)
        remaining_after = remaining[mask]
        bin_capacity_used = bins[mask]
        
        # Primary: minimize remaining space after placement
        # Secondary: among equal remaining, prefer bins with more used capacity
        # (this consolidates items and leaves cleaner bins)
        
        # Actually, let's just use -remaining_after with a tiny noise-free tie-breaker
        # based on bin index to make it deterministic but varied.
        # Better: use -remaining_after + 1e-6 * bin_capacity_used
        # This way, if two bins have the same remaining after placement,
        # we prefer the one that was already more full.
        
        scores[mask] = -remaining_after + 1e-6 * bin_capacity_used
    
    return scores
# EVOLVE-BLOCK-END
