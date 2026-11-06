import numpy as np

def _simulate_greedy_value(capacity, weights, values, exclude_idx):
    """
    Simulates the standard greedy density algorithm on the given arrays,
    excluding the item at exclude_idx.
    Returns the total value accumulated.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Create a list of available indices
    # Using a list of indices is faster for small N than boolean masks with np operations in a loop
    available = [i for i in range(n) if i != exclude_idx]
    
    total_value = 0.0
    current_cap = capacity
    
    while available:
        # Find index in 'available' with max density
        # We need to find the item in 'available' that has max values[i]/weights[i]
        # and also fits in current_cap.
        best_idx_in_avail = -1
        best_density = -1.0
        best_val = 0.0
        best_w = 0.0
        
        # Iterate to find best fitting item with max density
        # Optimization: Pre-compute densities? No, weights/values are fixed.
        for idx in available:
            w = weights[idx]
            if w > current_cap:
                continue
            v = values[idx]
            density = v / w
            if density > best_density:
                best_density = density
                best_idx_in_avail = idx
                best_val = v
                best_w = w
        
        if best_idx_in_avail == -1:
            break
            
        total_value += best_val
        current_cap -= best_w
        
        # Remove the chosen item from available
        available.remove(best_idx_in_avail)
        
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    best_total_val = -1.0
    best_idx = -1
    
    # Pre-compute densities for sorting in simulation? 
    # The simulation inside _simulate_greedy_value does a linear scan.
    # For N=100, linear scan is 100 ops.
    # The outer loop is 100.
    # The simulation loop runs at most 100 times (items picked).
    # So inner cost is 100 * 100 = 10,000 ops per candidate.
    # Total cost 100 * 10,000 = 1,000,000 ops.
    # This is acceptable.
    
    for i in range(n):
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
