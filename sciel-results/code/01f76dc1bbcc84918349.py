"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Uses a refined Next-Item Awareness heuristic:
    1. Estimate S_next using a blend of the current item and the median gap.
    2. Score bins based on remaining space, with a soft penalty for gaps
       that are too small to fit the estimated next item.
    3. This balances Best-Fit (tightness) with Next-Item fit (utility).
    """
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # 1. Identify valid bins
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return scores
        
    valid_bins = bins[valid_mask]
    new_rems = valid_bins - item
    
    # 2. Estimate S_next
    # Blend the current item (sample from distribution) with the median gap (system state)
    used_mask = (bins < 100) & (bins > 0)
    if np.any(used_mask):
        used_rems = bins[used_mask]
        median_gap = np.median(used_rems)
        # Weighted average: trust the current item more if gaps are small (stable system)
        # If gaps are large, the system is less stable, trust the item more?
        # Let's use a simple average.
        s_next = 0.5 * item + 0.5 * median_gap
    else:
        s_next = item
        
    s_next = max(s_next, 0.1)
    
    # 3. Calculate Scores
    # Base: Best Fit (minimize new_rems)
    base_scores = -new_rems
    
    # Penalty for "Dead" space (new_rems < s_next)
    # We want to penalize gaps that are too small to be useful for the next item.
    # Penalty magnitude scales with how much smaller the gap is than s_next.
    deficit = np.maximum(0, s_next - new_rems)
    
    # Penalty coefficient P
    # If P > 1, we prefer larger gaps among unsafe bins (closer to s_next).
    # If P is large, we strongly avoid unsafe bins.
    # P=2 seems reasonable to allow "tight unsafe" to beat "loose safe".
    P = 2.0
    penalty_scores = P * deficit
    
    valid_scores = base_scores - penalty_scores
    
    # 4. Assign scores
    scores[valid_mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
