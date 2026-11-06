import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s limit

def _fractional_bound(remaining_capacity, weights, values, sorted_indices_list, exclude_idx):
    """
    Computes the fractional knapsack value for the items in sorted_indices_list
    (sorted by density descending), excluding exclude_idx,
    given remaining_capacity.
    """
    if remaining_capacity <= 0:
        return 0.0
        
    total_val = 0.0
    cap = remaining_capacity
    
    for idx in sorted_indices_list:
        if idx == exclude_idx:
            continue
            
        w = weights[idx]
        v = values[idx]
        
        if w <= cap:
            total_val += v
            cap -= w
        else:
            # Take fraction
            if w > 0:
                total_val += v * (cap / w)
            cap = 0
            break
            
        if cap <= 0:
            break
            
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Check time budget
    if time.time() - _start_time > _TIMEOUT:
        # Fallback to simple greedy density
        best_idx = -1
        best_density = -1.0
        for i in range(n):
            w_i = remaining_weights[i]
            if w_i > remaining_capacity:
                continue
            density = remaining_values[i] / w_i
            if density > best_density:
                best_density = density
                best_idx = i
        return int(best_idx)
        
    best_total_val = -1.0
    best_idx = -1
    
    # Pre-compute densities for sorting in simulation
    densities = [remaining_values[i] / remaining_weights[i] if remaining_weights[i] > 0 else float('inf') for i in range(n)]
    
    # Sort candidate indices by density descending
    sorted_indices_list = list(range(n))
    sorted_indices_list.sort(key=lambda x: densities[x], reverse=True)
    
    for i in range(n):
        # Check time budget periodically
        if time.time() - _start_time > _TIMEOUT:
            break
            
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue
            
        # Value if we pick i
        v_i = remaining_values[i]
        
        # Value of the rest if we pick i (using fractional knapsack bound)
        future_val = _fractional_bound(remaining_capacity - w_i, 
                                       remaining_weights, 
                                       remaining_values, 
                                       sorted_indices_list, 
                                       i)
        
        total_val = v_i + future_val
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = i
            
    return int(best_idx)
