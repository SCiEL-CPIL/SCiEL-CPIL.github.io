"""Improved heuristic for ant colony offline bin packing using global waste minimization."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Since demands are integers passed as floats, we can work with integer residuals
    # for efficiency and precision in the waste calculation.
    # We'll assume the values are close to integers.
    d_int = np.rint(d).astype(int)
    
    # Precompute the best single-item fit for any residual space size 0..capacity
    # max_fit[r] = max(d_k) such that d_k <= r
    max_fit = np.zeros(int(cap) + 1, dtype=int)
    
    # Sort unique demands for efficient lookup
    unique_demands = np.unique(d_int)
    
    # Fill max_fit array
    # For each possible residual r, find the largest demand <= r
    # Since capacity is small (150), we can just iterate
    current_best = 0
    demand_idx = 0
    for r in range(int(cap) + 1):
        # Advance demand_idx to include all demands <= r
        while demand_idx < len(unique_demands) and unique_demands[demand_idx] <= r:
            current_best = unique_demands[demand_idx]
            demand_idx += 1
        max_fit[r] = current_best

    # Initialize heuristic matrix
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Compute heuristic for each pair
    # H[i, j] is attractiveness of placing j after i
    # If d[i] + d[j] <= cap, it fits.
    # Residual R = cap - d[i] - d[j]
    # Waste W = R - max_fit[R]
    # Heuristic = 1 / (W + eps)
    
    eps = 1e-6
    
    for i in range(n):
        di = d_int[i]
        for j in range(n):
            dj = d_int[j]
            total = di + dj
            if total <= int(cap):
                R = int(cap) - total
                # R is an integer between 0 and cap
                best_fill = max_fit[R]
                waste = R - best_fill
                H[i, j] = 1.0 / (waste + eps)
            else:
                H[i, j] = 1e-9
                
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
