"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_INFEASIBLE = -1e18
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity; higher = preferred.

    Strategy: Fragmentation-aware utility scoring with bimodal consolidation.
    1. Feasibility check.
    2. Calculate post-placement remaining space.
    3. Penalize creation of 'dead' fragments (20-50% of bin capacity).
    4. Bimodal bonus: Reward preserving large empty bins and filling small bins.
    5. Return scores, masking infeasible bins.
    """
    # Time guard: If we are close to the limit, return a simple fallback
    # to ensure we don't get killed mid-computation in a complex loop (though this is O(N)).
    # For this specific O(N) function, the time check is mostly for compliance with the directive,
    # but we can add a very fast path if the array is huge.
    
    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    if not np.any(feasible):
        return np.full_like(bins, _INFEASIBLE, dtype=float)

    # Initialize scores to infeasible
    scores = np.full_like(bins, _INFEASIBLE, dtype=float)
    
    # Only compute for feasible bins to save time and avoid NaNs
    idx = np.where(feasible)[0]
    if len(idx) == 0:
        return scores

    r = remaining[idx]  # Remaining space after placement
    # We assume bin capacity is 100 based on the task description.
    # If capacity is different, the ratios might need scaling, but 100 is the standard.
    C = 100.0
    
    # 1. Fragmentation Penalty
    # We want to avoid leaving a remainder that is too small to be useful for future items
    # but too large to be ignored. 
    # Specifically, if the item takes up a moderate fraction of the bin, the leftover
    # might be a "dead" fragment.
    # Let's define a penalty based on the ratio of the leftover space to the item size?
    # Or better, the ratio of the leftover space to the total capacity?
    
    # The inspiration mentions: "penalize bins based on the ratio of the item size to the remaining capacity"
    # Wait, the directive says: "penalizes bins based on the ratio of the item size to the remaining capacity, 
    # specifically targeting bins where the item is a moderate fraction (e.g., 20-50%) of the remaining space"
    
    # Let's re-read carefully: "targeting bins where the item is a moderate fraction (e.g., 20-50%) of the remaining space"
    # This implies we look at `item / bins` (before placement) or `item / (bins)`?
    # If item is 20-50% of the *current* bin capacity (remaining), it might create a bad fragment.
    
    # Let's use the ratio of the item to the *original* bin remaining (before placement).
    # ratio = item / bins[idx]
    # If ratio is in [0.2, 0.5], we apply a penalty.
    
    # However, a more robust measure of "bad fragment" is the *resulting* empty space.
    # If the resulting empty space is between, say, 5 and 30, it's often unusable for Weibull items 
    # (which typically have a mode around 10-20 and tail to 100).
    
    # Let's combine:
    # A. Utility of leftover space: 
    #    - If leftover is very small (< 5), it's "wasted" but the bin is full (good consolidation).
    #    - If leftover is moderate (5-30), it's "dead space" (bad).
    #    - If leftover is large (> 30), it's "viable space" (good, preserves optionality).
    
    # Let's create a smooth utility function for the leftover space `r`.
    # We want high scores for r in [0, 5] (full) and r in [50, 100] (empty/viable).
    # We want low scores for r in [10, 40] (fragmented).
    
    # Smooth bimodal score using Gaussians or similar.
    # Peak 1: r ~ 0 (Consolidation)
    # Peak 2: r ~ 80 (Preservation of large bins)
    
    # Let's use a combination of:
    # 1. -r (Tightest fit base)
    # 2. Bonus for r being very small (Full bin)
    # 3. Bonus for r being very large (Empty bin)
    # 4. Penalty for r being in the middle
    
    # Smooth penalty for "dead zone" [10, 40]
    # We can use a Gaussian centered at 25 with width 15.
    dead_zone_center = 25.0
    dead_zone_width = 15.0
    penalty = np.exp(-((r - dead_zone_center) ** 2) / (2 * dead_zone_width ** 2))
    
    # Bimodal bonuses
    # Bonus for empty bins (r > 60)
    empty_bonus = np.where(r > 60, (r - 60) / 40.0, 0.0) # Linear rise from 0 to 1 as r goes 60->100
    
    # Bonus for full bins (r < 10)
    full_bonus = np.where(r < 10, (10 - r) / 10.0, 0.0) # Linear drop from 1 to 0 as r goes 0->10
    
    # Base score: prefer tighter fit, but scaled down so bonuses can influence
    base_score = -r
    
    # Combine
    # We want the bimodal effect to be strong.
    score = base_score + 10.0 * (empty_bonus + full_bonus) - 20.0 * penalty
    
    scores[idx] = score
    
    return scores
# EVOLVE-BLOCK-END
