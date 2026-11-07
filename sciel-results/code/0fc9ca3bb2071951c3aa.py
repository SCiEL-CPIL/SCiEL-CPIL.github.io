"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ACO bin packing.
    
    The heuristic combines:
    1. Structural gap-utility: how well item j fits into the remaining space
       after item i (tighter fit = better, as it reduces wasted capacity)
    2. Data-driven co-occurrence prior: frequency with which pairs of items
       end up in the same bin across multiple randomized First-Fit runs
    
    Returns:
        n x n matrix where entry [i,j] is the attractiveness of placing item j
        right after item i in the same bin. Non-positive or non-finite values
        are replaced with 1e-9.
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # --- Structural gap-utility component ---
    # For pair (i, j): remaining space after placing i is capacity - demands[i]
    # If demands[j] fits, the "tightness" is 1 - demands[j]/remaining
    # Higher tightness means less wasted space, which is better for packing
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    remaining = capacity - demands[:, None]  # space left after item i
    # Tightness: 1 - (demand_j / remaining_space) when it fits
    # This is high when demand_j is close to filling the remaining space
    tightness = np.zeros((n, n), dtype=np.float64)
    mask = fits & (remaining > 0)
    tightness[mask] = 1.0 - demands[None, :][mask] / remaining[mask]
    # Ensure non-negative
    tightness = np.maximum(tightness, 0.0)
    # Pairs that don't fit get 0
    tightness[~fits] = 0.0
    
    # --- Data-driven co-occurrence prior ---
    # Run multiple randomized First-Fit simulations to build a co-occurrence matrix
    C = np.zeros((n, n), dtype=np.float64)
    num_runs = 50
    
    for seed in range(num_runs):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        
        bins = []
        current_bin = []
        current_sum = 0.0
        
        for idx in perm:
            d = demands[idx]
            if current_sum + d <= capacity:
                current_sum += d
                current_bin.append(idx)
            else:
                if current_bin:
                    bins.append(current_bin)
                current_bin = [idx]
                current_sum = d
        
        if current_bin:
            bins.append(current_bin)
        
        # Update co-occurrence matrix
        for bin_items in bins:
            for i in bin_items:
                for j in bin_items:
                    if i != j:
                        C[i, j] += 1
    
    # Normalize co-occurrence to [0, 1] range
    max_C = C.max() if C.max() > 0 else 1.0
    C_normalized = C / max_C
    
    # --- Combine the two signals ---
    # Weight the structural and data-driven components
    # The structural signal captures local tightness, the co-occurrence captures global structure
    alpha = 0.5  # weight for structural gap-utility
    beta = 0.5   # weight for co-occurrence prior
    
    H = alpha * tightness + beta * C_normalized
    
    # Ensure diagonal is 0 (an item can't follow itself in a meaningful way for ACO)
    np.fill_diagonal(H, 0.0)
    
    # Replace non-positive or non-finite values with 1e-9
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
