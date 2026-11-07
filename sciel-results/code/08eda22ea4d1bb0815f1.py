"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    start_time = time.time()
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pairwise sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Validity mask: item j can follow item i if their sum <= capacity
    valid = pair_sum <= capacity
    
    # Calculate waste for valid pairs
    waste = capacity - pair_sum
    waste = np.where(valid, waste, 0.0)
    
    # Fit heuristic: prefer less waste with moderate discrimination.
    # Using 1/(1+waste)^1.5 gives stronger penalty for larger waste than power 1,
    # but less aggressive than power 2, balancing tightness with flexibility.
    fit = 1.0 / (1.0 + waste) ** 1.5
    
    # Scarcity heuristic: prefer items that have fewer compatible partners
    # compat_count[j] is the number of items i such that d_i + d_j <= capacity
    compat_count = valid.sum(axis=0).astype(float)
    
    # Use inverse for scarcity signal (linear inverse is more stable than squared)
    scarcity = 1.0 / np.maximum(compat_count, 1.0)
    
    # Forward flexibility: after placing i then j, how many items k can still fit?
    # remaining space = waste[i,j]
    # forward_flex[i,j] = count of k where d[k] <= waste[i,j]
    sorted_demands = np.sort(demands)
    forward_flex = np.zeros((n, n), dtype=float)
    for j in range(n):
        rem = capacity - demands[j] - demands  # shape (n,)
        # Count how many sorted_demands are <= rem
        forward_flex[:, j] = np.searchsorted(sorted_demands, rem, side='right')
    
    # Normalize forward flexibility to [0, 1]
    forward_flex_norm = forward_flex / max(n, 1)
    
    # Combine heuristics:
    # eta[i, j] = fit[i,j] * scarcity[j] * (0.5 + 0.5 * forward_flex_norm[i,j])
    # The forward flex term rewards placements that leave room for more future items,
    # while fit rewards tight packing and scarcity rewards placing constrained items.
    eta = fit * scarcity[None, :] * (0.5 + 0.5 * forward_flex_norm)
    
    # Set invalid entries to 1e-9
    eta = np.where(valid, eta, 1e-9)
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Simple deadline guard
    if time.time() - start_time > 250:
         # Fallback to seed-like heuristic if somehow slow (unlikely)
         pair = demands[:, None] + demands[None, :]
         return np.where(pair <= capacity, pair / capacity, 1e-9)
         
    return eta
# EVOLVE-BLOCK-END
