import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # Sort indices by size descending (stable sort for determinism)
    sorted_indices = np.argsort(-demands, kind='mergesort')
    
    # Run First-Fit-Decreasing
    bins = []  # list of lists of original indices
    bin_remaining = []  # remaining capacity for each bin
    
    for idx in sorted_indices:
        size = demands[idx]
        placed = False
        for b in range(len(bins)):
            if size <= bin_remaining[b]:
                bins[b].append(idx)
                bin_remaining[b] -= size
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_remaining.append(capacity - size)
    
    # Build sparse heuristic matrix: high value only for consecutive pairs in FFD bins
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    for bin_items in bins:
        for k in range(len(bin_items) - 1):
            i = bin_items[k]
            j = bin_items[k + 1]
            H[i, j] = 1.0
    
    return H
