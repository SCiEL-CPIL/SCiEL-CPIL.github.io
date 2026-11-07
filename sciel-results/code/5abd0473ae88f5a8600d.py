"""
Bootstrap spawn: Direct approach using theoretical bin packing insights.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy:
    Combined fill-ratio, gap-filling, and complementarity heuristic:
    1. Fill ratio term: (pair_sums / capacity)^alpha rewards high utilization.
    2. Gap-filling term: soft-min of remaining capacity rewards pairs that leave
       minimal unused space (smooth surrogate for the non-smooth min objective).
    3. Complementarity term: rewards pairs with moderate size differences.
    
    The product of these terms creates a multi-dimensional attractiveness score.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Compute pairwise sums and absolute differences
    pair_sums = d[:, None] + d[None, :]
    pair_diffs = np.abs(d[:, None] - d[None, :])
    
    # Determine which pairs fit in a bin
    fits = pair_sums <= capacity
    
    # Fill ratio: (pair_sums / capacity)^alpha
    alpha = 20.0
    ratios = pair_sums / capacity
    fill_term = np.power(ratios, alpha)
    
    # Gap-filling term: smooth surrogate for minimizing remaining space
    # remaining = capacity - pair_sums (when fits, else large)
    remaining = np.where(fits, capacity - pair_sums, capacity)
    # Soft-min: use exp(-remaining / scale) to reward small remaining space
    # Scale controls how sharply we prefer tight fits
    scale = 10.0
    gap_term = np.exp(-remaining / scale)
    
    # Complementarity term: 1 + beta * (normalized size difference)
    # Moderate differences are best; too similar wastes potential, too different may not fit
    max_diff = 79.0  # Since demands are in [20, 99]
    beta = 0.5
    normalized_diff = pair_diffs / max_diff
    # Use a peaked function: max at moderate differences
    complementarity = 1.0 + beta * (normalized_diff * (1.0 - normalized_diff) * 4.0)
    
    # Combine: multiplicative combination
    H_fit = fill_term * gap_term * complementarity
    
    # Mask out non-fitting pairs
    H = np.where(fits, H_fit, 1e-9)
    
    # Ensure no non-finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # The task says "Heuristic entries that are not positive become 1e-9"
    H = np.where(H > 0, H, 1e-9)

    return H

# EVOLVE-BLOCK-END
