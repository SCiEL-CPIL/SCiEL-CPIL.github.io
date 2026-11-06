import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 1e-12:
        return -1
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Convert to numpy arrays for convenience
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Filter feasible items
    feasible_mask = weights <= remaining_capacity + 1e-12
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Compute densities for all items
    densities = np.zeros(n)
    for i in range(n):
        if weights[i] > 0:
            densities[i] = values[i] / weights[i]
        else:
            densities[i] = float('inf') if values[i] > 0 else 0.0
    
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    best_score = -1.0
    best_idx = -1
    
    # For each feasible candidate i, compute score = v_i + greedy_01_knapsack_value(remaining, capacity - w_i, excluding i)
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        new_cap = remaining_capacity - w_i
        
        if new_cap < -1e-12:
            continue
        
        # Compute greedy 0/1 knapsack value on all items except i, with capacity new_cap
        # Iterate through sorted_indices, skip i
        greedy_val = 0.0
        current_cap = new_cap
        for idx in sorted_indices:
            if idx == i:
                continue
            if current_cap <= 1e-12:
                break
            w_idx = weights[idx]
            v_idx = values[idx]
            if w_idx <= current_cap + 1e-12:
                greedy_val += v_idx
                current_cap -= w_idx
            # In 0/1 greedy, if it doesn't fit, we just skip it (unlike fractional)
        
        total_score = v_i + greedy_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
        
        # Time check
        if time.time() > deadline:
            break
    
    # If no valid index found via greedy scoring, fall back to best density
    if best_idx < 0:
        # Pick the feasible item with highest density
        best_density = -1.0
        for i in feasible_indices:
            if densities[i] > best_density:
                best_density = densities[i]
                best_idx = i
    
    return int(best_idx) if best_idx >= 0 else -1
