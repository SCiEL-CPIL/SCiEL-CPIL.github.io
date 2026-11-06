import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Find indices of feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    best_value = -1
    best_idx = -1
    
    # For each feasible item, simulate the greedy completion
    for idx in feasible_indices:
        w = remaining_weights[idx]
        v = remaining_values[idx]
        
        # Start with this item
        curr_cap = remaining_capacity - w
        curr_val = v
        
        # Simulate the rest of the greedy process
        # We need to pick the best density from the remaining items
        # that fit in curr_cap
        
        # Create a list of other items
        # To avoid copying arrays repeatedly, we can just iterate and check
        # But we need to simulate a sequence of picks.
        
        # Let's create a temporary list of (density, weight, value) for others
        # Or just repeatedly find the max density feasible item
        
        # Efficient simulation:
        # We have n-1 items. We pick the best density one that fits, then the next, etc.
        
        # To make this fast, we can sort the other items by density once per idx?
        # That's O(N log N) per idx, so O(N^2 log N) total. 100^2 * 7 = 70,000 ops. Fine.
        
        others_indices = np.setdiff1d(np.arange(n), [idx])
        
        if len(others_indices) > 0:
            other_weights = remaining_weights[others_indices]
            other_values = remaining_values[others_indices]
            
            # Calculate densities
            with np.errstate(divide='ignore', invalid='ignore'):
                densities = other_values / other_weights
            # Handle zero weight (shouldn't happen, but safe)
            densities = np.where(other_weights > 0, densities, np.inf)
            
            # Sort by density descending
            sort_order = np.argsort(-densities)
            
            for j in sort_order:
                w_j = other_weights[j]
                if w_j <= curr_cap:
                    curr_cap -= w_j
                    curr_val += other_values[j]
                # If it doesn't fit, we skip it in greedy? 
                # Standard greedy: if it doesn't fit, we just don't pick it. 
                # We don't stop, we continue to check smaller items? 
                # No, standard greedy by density usually implies we pick the best that fits.
                # If the best density item doesn't fit, does the second best fit? 
                # Yes, it might. So we must check all.
        
        if curr_val > best_value:
            best_value = curr_val
            best_idx = idx
            
    return int(best_idx)
