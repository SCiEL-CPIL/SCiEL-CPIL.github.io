import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 115.0  # Safety margin from 120s

def _lp_bound(capacity, weights, values, available_mask):
    """
    Calculate the LP relaxation upper bound for the knapsack problem
    with given capacity, weights, values, and a boolean mask of available items.
    """
    if capacity <= 0:
        return 0.0
    
    # Get indices of available items
    indices = np.where(available_mask)[0]
    if len(indices) == 0:
        return 0.0
    
    # Sort indices by value density (v/w) descending
    # We need to handle division by zero, but weights should be > 0
    densities = values[indices] / weights[indices]
    sorted_indices = indices[np.argsort(densities)[::-1]]
    
    total_value = 0.0
    rem_cap = capacity
    
    for idx in sorted_indices:
        w = weights[idx]
        v = values[idx]
        if w <= rem_cap:
            total_value += v
            rem_cap -= w
        else:
            # Take fraction
            total_value += v * (rem_cap / w)
            break
            
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback to simple density greedy if out of time
        # Find best density item
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        densities = np.where(feasible, remaining_values / remaining_weights, 0)
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    best_score = -1
    best_index = -1
    
    # Pre-compute a mask for all items
    # For each candidate i, we need the LP bound of the others.
    
    # Optimization: 
    # The LP bound calculation involves sorting. 
    # We can pre-sort all items by density.
    # Let sorted_all = argsort(densities)[::-1]
    # For a candidate i, the available items are all except i.
    # The LP bound is the sum of values in sorted_all order (skipping i) until capacity is filled.
    
    # Let's implement this optimized version.
    densities = remaining_values / remaining_weights
    # Handle potential division by zero if any weight is 0 (unlikely in knapsack)
    # If weight is 0, density is inf.
    # np.sort handles inf.
    
    sorted_order = np.argsort(densities)[::-1]
    
    # We iterate through feasible candidates.
    # To speed up LP bound, we can pre-compute prefix sums of values and weights 
    # in the sorted order? 
    # No, because we skip item i. 
    # But N=100, so O(N) per candidate is O(N^2) total, which is 10,000 ops. 
    # This is very fast.
    
    for i in feasible_indices:
        # Construct available mask: all True except i
        # Instead of creating a new mask each time, we can just loop through sorted_order
        # and skip i.
        
        cap_rem = remaining_capacity - remaining_weights[i]
        if cap_rem < 0:
            continue # Should not happen as i is feasible
        
        lp_val = 0.0
        rem_cap = cap_rem
        
        for idx in sorted_order:
            if idx == i:
                continue
            w = remaining_weights[idx]
            v = remaining_values[idx]
            if w <= rem_cap:
                lp_val += v
                rem_cap -= w
            else:
                if rem_cap > 0:
                    lp_val += v * (rem_cap / w)
                break
        
        total_score = remaining_values[i] + lp_val
        
        if total_score > best_score:
            best_score = total_score
            best_index = i
            
    if best_index == -1:
        # Fallback if something went wrong
        return int(feasible_indices[0])
        
    return int(best_index)
