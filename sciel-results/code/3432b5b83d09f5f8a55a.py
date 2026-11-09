import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter out items that don't fit? 
    # The problem says "remaining (feasible) arrays". 
    # "feasible" might mean they all fit individually? 
    # "picks the next item to add... while building 0/1 knapsack solutions"
    # Usually, the caller removes the picked item and updates capacity.
    # If an item doesn't fit, we shouldn't pick it.
    # But the input is "remaining_weights". If an item is in the array, is it feasible?
    # The prompt says "index into the remaining (feasible) arrays". 
    # This implies the arrays passed in only contain items that are currently feasible (weight <= remaining_capacity)?
    # Or does it mean the arrays are the set of items not yet picked, and we must check feasibility?
    # "remaining (feasible) arrays" is slightly ambiguous. 
    # Standard interpretation: The arrays contain all unpicked items. We must ensure the picked item fits.
    # If an item doesn't fit, we skip it.
    # If NO items fit, return -1.
    
    # Check if any item fits
    if np.any(weights > remaining_capacity):
        # It's possible some don't fit. We only consider those that do.
        pass # We will handle infeasibility in the candidate loop.

    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0)
    
    # Get indices sorted by density descending
    # Tie-breaker: higher value first? 
    # Let's just use density.
    sorted_indices = np.argsort(-densities)
    
    # We look at the top K candidates
    K = min(50, n)
    
    best_total_value = -1
    best_idx = -1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0 # 1 second per call is generous. 800 calls * 1s = 800s > 120s. 
    # Wait, 120s total. 8 instances. 
    # If we use 1s per call, and there are ~100 calls per instance, that's 100s per instance. 
    # 8 * 100s = 800s. Too long.
    # We need to be faster.
    # The loop is 5000 iterations. In Python, 5000 iterations is ~0.1ms - 1ms.
    # 800 calls * 1ms = 0.8s. This is safe.
    # So we don't really need a tight deadline guard, but we should have one to prevent infinite loops or bugs.
    
    for idx in sorted_indices[:K]:
        i = int(idx)
        if weights[i] > remaining_capacity:
            continue
            
        # Simulate picking i
        curr_cap = remaining_capacity - weights[i]
        total_val = values[i]
        
        # Greedily fill the rest
        for j in sorted_indices:
            j = int(j)
            if j == i:
                continue
            if weights[j] <= curr_cap:
                curr_cap -= weights[j]
                total_val += values[j]
                
        if total_val > best_total_value:
            best_total_value = total_val
            best_idx = i

    return best_idx
