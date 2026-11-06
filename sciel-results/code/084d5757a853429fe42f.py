import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 1e-12:
        return -1

    start_time = time.time()
    deadline = start_time + 110.0

    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)

    feasible_mask = weights <= remaining_capacity + 1e-12
    feasible_indices = np.where(feasible_mask)[0]

    if len(feasible_indices) == 0:
        return -1

    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Calculate densities
    densities = np.zeros(n)
    for i in range(n):
        if weights[i] > 0:
            densities[i] = values[i] / weights[i]
        else:
            densities[i] = float('inf') if values[i] > 0 else 0.0

    # Sort indices by density (descending)
    sorted_indices = np.argsort(-densities)
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    
    # Map original index to its rank in sorted order
    rank_of = np.zeros(n, dtype=int)
    for rank, idx in enumerate(sorted_indices):
        rank_of[idx] = rank

    # Prefix sums of weights and values in sorted order
    prefix_weights = np.zeros(n + 1)
    prefix_values = np.zeros(n + 1)
    for i in range(n):
        prefix_weights[i + 1] = prefix_weights[i] + sorted_weights[i]
        prefix_values[i + 1] = prefix_values[i] + sorted_values[i]

    best_score = -1.0
    best_idx = -1

    # For each feasible item, estimate the total value if we pick it
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        new_cap = remaining_capacity - w_i
        
        if new_cap < -1e-12:
            continue

        # Compute fractional knapsack value for S \ {i} with capacity new_cap
        # Use binary search with prefix sums, adjusting for removed item i
        r_i = rank_of[i]
        
        # We want to find the largest number of full items k from S\{i} 
        # such that their total weight <= new_cap.
        # 
        # Case 1: If we take k items from positions 0..k-1 in sorted order,
        #   and r_i >= k: sum = prefix_weights[k], count = k
        #   and r_i < k: sum = prefix_weights[k] - w_i (since item i is in first k), 
        #   but wait - if r_i < k, then among first k positions, one is i, so 
        #   the actual items taken are k-1 items from first k positions.
        #   Actually, "first k items in S\{i}" means:
        #   Take items from sorted order, skipping i, until we have k items.
        #   If r_i >= k: items are at positions 0..k-1, sum = prefix_weights[k]
        #   If r_i < k: items are at positions 0..k (skipping r_i), so k items from k+1 positions
        #   sum = prefix_weights[k+1] - w_i
        #
        # So define: sum_first_k_excl_i(k) = 
        #   prefix_weights[k] if r_i >= k
        #   prefix_weights[k+1] - w_i if r_i < k
        #
        # We want max k such that sum_first_k_excl_i(k) <= new_cap
        # Then fractional value = prefix_values_excl_i(k) + fraction of next item
        
        # Binary search for max k where sum_first_k_excl_i(k) <= new_cap
        lo, hi = 0, n - 1  # k ranges from 0 to n-1 (can't take more than n-1 items from S\{i})
        best_k = 0
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if r_i >= mid:
                s = prefix_weights[mid]
            else:
                s = prefix_weights[mid + 1] - w_i
            
            if s <= new_cap + 1e-12:
                best_k = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        # Compute fractional value for S\{i} with capacity new_cap
        # Full items: best_k items
        if r_i >= best_k:
            full_val = prefix_values[best_k]
            full_w = prefix_weights[best_k]
            # Next item to consider is at position best_k in sorted order
            next_pos = best_k
        else:
            # r_i < best_k, so we took items from positions 0..best_k, skipping r_i
            full_val = prefix_values[best_k + 1] - v_i
            full_w = prefix_weights[best_k + 1] - w_i
            # Next item to consider is at position best_k + 1 in sorted order
            next_pos = best_k + 1
        
        rem_cap = new_cap - full_w
        frac_val = full_val
        
        if rem_cap > 1e-12 and next_pos < n:
            next_idx = sorted_indices[next_pos]
            if next_idx != i:
                w_next = weights[next_idx]
                v_next = values[next_idx]
                if w_next > 0:
                    frac_val += min(1.0, rem_cap / w_next) * v_next
                else:
                    frac_val += v_next
        
        total_score = v_i + frac_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
        
        if time.time() > deadline:
            break
    
    if best_idx < 0:
        # Fallback: highest density
        best_density = -1.0
        for i in feasible_indices:
            if densities[i] > best_density:
                best_density = densities[i]
                best_idx = i
    
    return int(best_idx) if best_idx >= 0 else -1
