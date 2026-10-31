"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback to simple Best Fit
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)

    # Calculate the waste (remaining space) for each valid bin
    waste = bins - item
    
    # Structural Change: Entropy-Regularized Packing
    # The current approach uses a fixed or item-size-dependent penalty on waste.
    # This new approach models the "state" of the bin pool.
    # We want to maintain a diverse set of remaining capacities to handle future items.
    # If the remaining capacities are all very similar (low entropy), we risk fragmentation.
    # If they are diverse (high entropy), we have flexibility.
    # 
    # Strategy:
    # 1. Compute the entropy of the valid remaining capacities (normalized).
    # 2. If entropy is low (bins are homogeneous), we should prefer bins that increase diversity (i.e., move away from the mean/median).
    # 3. If entropy is high, we can afford to be more greedy (Best Fit) to consolidate.
    #
    # However, a simpler and more robust structural change is to use a "Gap" based heuristic.
    # Instead of just minimizing waste, we look at the gap between the item and the bin.
    # Weibull distributions have a specific shape parameter. 
    # A key insight in bin packing is that "Best Fit" often fails because it creates many small scraps.
    # "Worst Fit" keeps bins empty but uses more bins.
    # "First Fit Decreasing" is optimal offline but not online.
    #
    # Let's try a "Relative Best Fit" with a dynamic threshold.
    # If the best available bin has waste W_best.
    # If W_best is very small, it's a perfect fit. Take it.
    # If W_best is large, maybe we should take a bin with even larger waste? No, that's Worst Fit.
    #
    # Alternative: **Harmonic Mean of Waste**
    # This is a different mathematical family.
    # Instead of sum or mean, we consider the harmonic mean of the wastes of the bins we've already used? No, we only see current state.
    #
    # Let's try: **Distance to a Target Remaining Capacity**
    # For Weibull items, the optimal remaining capacity might not be 0.
    # If we assume a target remaining capacity T (e.g., 10% of bin capacity = 10),
    # we score bins by how close their post-item waste is to T.
    # Score = -|waste - T|
    # This is a "Target Fit" strategy.
    #
    # Why might this work?
    # If we always leave ~10 space, we can fit small items later.
    # If we leave 0 space, we can't fit anything else.
    # If we leave 50 space, we're wasting space that could have held two 25s.
    #
    # Let's estimate T from the current item size.
    # If item is large, T should be small (we can't expect much left).
    # If item is small, T can be larger.
    # Let T = max(5, 0.1 * (100 - item)) ? Or just a fixed constant?
    #
    # Let's try a fixed target of 10.
    # Score = -|waste - 10|
    
    # But what if no bin has waste near 10?
    # Then we pick the one closest to 10.
    # If all wastes are < 10, we pick the one with max waste (closest to 10 from below).
    # If all wastes are > 10, we pick the one with min waste (closest to 10 from above).
    #
    # This is structurally different from monotonic Best Fit or Worst Fit.
    # It introduces a "valley" in the scoring function.
    
    target_waste = 10.0
    scores = -np.abs(waste - target_waste)
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
