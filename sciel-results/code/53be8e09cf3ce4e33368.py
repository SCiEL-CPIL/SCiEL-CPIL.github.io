"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands, dtype=np.float64)
    
    # 1. Compute Gap Matrix
    # gap[i, j] = capacity - (demands[i] + demands[j])
    # If gap >= 0, items i and j can fit together.
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum
    
    # Valid mask: gap >= 0
    valid = gap >= 0
    
    # 2. Base Heuristic: Tight Fit Reward
    # We want to reward placing item j if it fits tightly after i.
    # Formula: d_j / (1 + gap) for valid pairs.
    # This encourages placing larger items (d_j) into tighter gaps.
    # For invalid pairs, set to 1e-9.
    
    # Avoid division by zero or negative numbers in the base calculation
    # We only calculate for valid pairs effectively, but vectorized is faster.
    # Use np.where to handle invalid cases safely.
    # If gap < 0, we want 1e-9.
    # If gap >= 0, we want demands[j] / (1 + gap).
    
    # To avoid warnings/errors, we compute on a safe array
    safe_gap = np.where(valid, gap, 1.0)
    base_heuristic = demands[None, :] / (1.0 + safe_gap)
    
    # Mask invalid pairs
    H = np.where(valid, base_heuristic, 1e-9)
    
    # 3. FFD Structural Prior
    # Run First Fit Decreasing to get a reference packing.
    # This identifies which pairs of items are "compatible" (end up in same bin).
    
    # Sort indices by demand descending
    sorted_indices = np.argsort(demands)[::-1]
    sorted_demands = demands[sorted_indices]
    
    bins = []
    bin_sums = []
    
    for idx in sorted_indices:
        d = demands[idx]
        placed = False
        for k in range(len(bins)):
            if bin_sums[k] + d <= capacity:
                bins[k].append(idx)
                bin_sums[k] += d
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_sums.append(d)
            
    # Create a matrix indicating if two items are in the same FFD bin
    # This is a symmetric matrix.
    ffd_same_bin = np.zeros((n, n), dtype=np.float64)
    for b in bins:
        for i_idx in b:
            for j_idx in b:
                if i_idx != j_idx:
                    ffd_same_bin[i_idx, j_idx] = 1.0
    
    # 4. Combine Base Heuristic and FFD Prior
    # Boost pairs that were in the same FFD bin.
    # H_final = H_base * (1 + alpha * ffd_same_bin)
    # Let's use alpha = 5.0 to strongly favor FFD-compatible pairs.
    alpha = 5.0
    H = H * (1.0 + alpha * ffd_same_bin)
    
    # 5. Cleanup
    # Ensure diagonal is 1e-9 (placing item after itself is not allowed/meaningful)
    np.fill_diagonal(H, 1e-9)
    
    # Ensure no non-finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Ensure minimum value is 1e-9 for any non-positive or zero values that might arise
    # (Though base_heuristic is positive for valid, and 1e-9 for invalid)
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
