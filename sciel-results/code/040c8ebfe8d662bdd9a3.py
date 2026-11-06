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
    # rank_of[i] = index in sorted_indices where sorted_indices[rank] == i
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
        # We iterate through the sorted items, skipping i
        # Optimization: Use binary search to find the cut point
        
        frac_val = 0.0
        current_cap = new_cap
        
        # We need to find the number of full items we can take from S \ {i}
        # Let k be the number of full items.
        # The weights of the first k items in S \ {i} must be <= new_cap.
        
        # We can binary search on the prefix sums of the full set,
        # but we have to adjust for the removed item i.
        
        # Let r_i be the rank of i in the sorted order.
        r_i = rank_of[i]
        
        # We want to find the largest k such that sum_{j in S\{i}, j <= k} w_j <= new_cap
        # If k < r_i: sum is prefix_weights[k]
        # If k >= r_i: sum is prefix_weights[k+1] - w_i
        
        # We can perform a binary search on k from 0 to n-1
        lo, hi = 0, n - 1
        best_k = -1
        
        # Helper function to get sum of weights of first k items in S \ {i}
        # Note: "first k items in S \ {i}" means the k items with highest density, excluding i.
        # If we take the first k items from the full sorted list:
        # If r_i >= k, the set is {0, ..., k-1}. Sum = prefix_weights[k].
        # If r_i < k, the set is {0, ..., k-1} \ {r_i}. This is k-1 items.
        # Wait, the definition of "first k items in S \ {i}" is tricky.
        
        # Let's just iterate. With n=100, iteration is fast enough.
        # The previous code did this.
        # To speed it up, we can break early if current_cap is small.
        
        for rank, idx in enumerate(sorted_indices):
            if idx == i:
                continue
            if current_cap <= 1e-12:
                break
            w_idx = weights[idx]
            v_idx = values[idx]
            if w_idx <= current_cap + 1e-12:
                frac_val += v_idx
                current_cap -= w_idx
            else:
                if w_idx > 0:
                    frac_val += (current_cap / w_idx) * v_idx
                current_cap = 0.0
                break
        
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
