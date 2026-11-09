import numpy as np
import time

def heuristics(demands, capacity):
    n = len(demands)
    # Deadline guard
    start_time = time.time()
    deadline = 290.0  # Leave 10s margin
    
    # 1. Compute First-Fit Decreasing (FFD) solution
    # Sort indices by demand in decreasing order
    sorted_indices = np.argsort(demands)[::-1]
    
    # FFD Packing
    bins = []
    bin_fill = []
    item_to_bin = [-1] * n
    
    for idx in sorted_indices:
        d = demands[idx]
        placed = False
        for b in range(len(bins)):
            if bin_fill[b] + d <= capacity:
                bin_fill[b] += d
                item_to_bin[idx] = b
                placed = True
                break
        if not placed:
            bins.append(d)
            bin_fill.append(d)
            item_to_bin[idx] = len(bins) - 1
            
    # 2. Construct Heuristic Matrix
    # Initialize with very low value for all pairs
    H = np.full((n, n), 1e-9, dtype=float)
    
    # Identify pairs that fit
    # d_i + d_j <= capacity
    d_col = demands[:, None]
    d_row = demands[None, :]
    fits = (d_col + d_row) <= capacity
    
    # Set base attractiveness for fitting pairs
    H[fits] = 1.0
    
    # Boost pairs that are in the same FFD bin
    # We can iterate through bins and set H[i][j] and H[j][i] for all pairs in the bin
    # Or use a mask
    # Since N is small, we can just loop through items
    for i in range(n):
        bin_i = item_to_bin[i]
        for j in range(n):
            if i == j:
                continue
            if item_to_bin[j] == bin_i:
                # They are in the same FFD bin
                # Ensure they fit (they must, by FFD construction)
                H[i, j] = 10.0
                
    # Optional: Time check
    if time.time() > deadline:
        return H

    return H
