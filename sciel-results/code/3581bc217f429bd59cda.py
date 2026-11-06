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

    densities = np.zeros(n)
    for i in range(n):
        if weights[i] > 0:
            densities[i] = values[i] / weights[i]
        else:
            densities[i] = float('inf') if values[i] > 0 else 0.0

    sorted_indices = np.argsort(-densities)

    # Precompute prefix sums of values and weights in density order
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    
    # For fractional knapsack, we need to quickly compute the value of
    # filling capacity C with items in density order (excluding a specific item)
    # We can use prefix sums and binary search
    
    prefix_weights = np.zeros(n + 1)
    prefix_values = np.zeros(n + 1)
    for i in range(n):
        prefix_weights[i + 1] = prefix_weights[i] + sorted_weights[i]
        prefix_values[i + 1] = prefix_values[i] + sorted_values[i]
    
    best_score = -1.0
    best_idx = -1
    
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        new_cap = remaining_capacity - w_i
        
        if new_cap < -1e-12:
            continue
        
        # Compute fractional knapsack value on all items except i, with capacity new_cap
        # We iterate through sorted_indices, skipping i
        # To speed this up, we can precompute the prefix sums and use them
        
        frac_val = 0.0
        current_cap = new_cap
        pos_in_sorted = -1
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
                frac = current_cap / w_idx if w_idx > 0 else 0.0
                frac_val += frac * v_idx
                current_cap = 0.0
                break
        
        total_score = v_i + frac_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
        
        if time.time() > deadline:
            break
    
    if best_idx < 0:
        best_density = -1.0
        for i in feasible_indices:
            if densities[i] > best_density:
                best_density = densities[i]
                best_idx = i
    
    return int(best_idx) if best_idx >= 0 else -1
