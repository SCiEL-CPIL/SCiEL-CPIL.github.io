"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    w = np.array(remaining_weights, dtype=float)
    v = np.array(remaining_values, dtype=float)
    
    # Calculate densities
    # Handle zero weights: if w=0 and v>0, density is inf. If w=0 and v=0, density 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(w > 0, v / w, np.where(v > 0, np.inf, 0.0))

    # Indices of items that fit in current capacity
    feasible_mask = w <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    best_score = -1
    best_idx = -1

    # Pre-sort all indices by density descending to make filling faster?
    # Not directly useful because the subset changes (rem_cap changes).
    # But we can iterate.
    
    for i in feasible_indices:
        if w[i] > remaining_capacity:
            continue
            
        current_val = v[i]
        rem_cap = remaining_capacity - w[i]
        
        # Find items j != i that fit in rem_cap
        # We can filter the full list
        # To speed up, we can just iterate all n items
        # Since n=100, this is fine.
        
        # Collect candidates for greedy fill
        # We want them sorted by density.
        # Instead of sorting every time, we can create a list of (density, index) 
        # and filter, then sort.
        
        candidates = []
        for j in range(n):
            if j == i:
                continue
            if w[j] <= rem_cap:
                candidates.append((densities[j], j))
                
        # Sort by density descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        greedy_val = 0
        curr_cap = rem_cap
        for density, j in candidates:
            if w[j] <= curr_cap:
                greedy_val += v[j]
                curr_cap -= w[j]
                
        total_val = current_val + greedy_val
        
        if total_val > best_score:
            best_score = total_val
            best_idx = i

    if best_idx == -1:
        return -1
        
    return int(best_idx)
# EVOLVE-BLOCK-END
