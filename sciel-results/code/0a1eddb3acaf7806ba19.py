import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s limit

def _simulate_greedy_value(capacity, weights, values, exclude_idx):
    """
    Simulates the standard greedy density algorithm on the given arrays,
    excluding the item at exclude_idx.
    Returns the total value accumulated.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Pre-compute densities for all items once
    densities = [values[i] / weights[i] if weights[i] > 0 else float('inf') for i in range(n)]
    
    # Create a list of available indices, sorted by density descending
    # This allows us to pick the best fitting item more efficiently
    available = [i for i in range(n) if i != exclude_idx]
    # Sort by density descending
    available.sort(key=lambda x: densities[x], reverse=True)
    
    total_value = 0.0
    current_cap = capacity
    
    while available:
        # Find the first item in the sorted list that fits in current_cap
        best_idx_in_avail = -1
        best_val = 0.0
        best_w = 0.0
        
        for idx in available:
            w = weights[idx]
            if w <= current_cap:
                best_idx_in_avail = idx
                best_val = values[idx]
                best_w = w
                break
        
        if best_idx_in_avail == -1:
            break
            
        total_value += best_val
        current_cap -= best_w
        
        # Remove the chosen item from available
        available.remove(best_idx_in_avail)
        
    return total_value


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
    
    # Sort candidate indices by density descending to potentially find good solutions earlier
    candidates = list(range(n))
    candidates.sort(key=lambda x: densities[x], reverse=True)
    
    for i in candidates:
        # Check time budget periodically
        if time.time() - _start_time > _TIMEOUT:
            break
            
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue
            
        # Value if we pick i
        v_i = remaining_values[i]
        
        # Value of the rest if we pick i (using greedy density on the rest)
        # We pass the original arrays and the index to exclude
        future_val = _simulate_greedy_value(remaining_capacity - w_i, 
                                            remaining_weights, 
                                            remaining_values, 
                                            i)
        
        total_val = v_i + future_val
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = i
            
    return int(best_idx)
