import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Pre-sort all indices by value density (value / weight) descending
    # We only care about feasible items for the initial candidate list, 
    # but the "fill" can use any item that fits in the remaining capacity 
    # after the candidate is picked. However, items that don't fit in 
    # remaining_capacity now might fit later? No, capacity only decreases.
    # So any item with weight > remaining_capacity is useless.
    # Actually, the input `remaining_weights` and `remaining_values` 
    # likely contain only items that were feasible at the start or 
    # are the current pool. The problem statement says "remaining (feasible) arrays".
    # This implies all items in the arrays are feasible? 
    # "index into the remaining (feasible) arrays". 
    # This suggests the caller filters out infeasible items? 
    # Or does it mean we should filter?
    # The seed code does `np.argmax(remaining_values / remaining_weights)`.
    # If an item is infeasible (weight > capacity), it shouldn't be picked.
    # The prompt says "picks the next item to add... index into the remaining (feasible) arrays".
    # This phrasing is ambiguous. Does the input contain infeasible items?
    # Usually in these iterative heuristics, the state is maintained by the caller.
    # If the caller passes infeasible items, we must filter them.
    # Let's assume we must filter to be safe.
    
    # We already computed feasible_mask.
    # Let's work with the full arrays but mask out infeasible ones during evaluation.
    
    # Calculate densities
    # Avoid division by zero, though weights should be positive
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(remaining_weights > 0, remaining_values / remaining_weights, 0)
        
    # Sort indices by density descending
    # We only need to consider feasible items as candidates for the FIRST pick.
    # For the "fill" part, we consider all items (feasible or not? No, only those that fit in the reduced capacity).
    
    # Let's stick to the feasible indices for candidates.
    # For the fill, we can use the global sorted list, but we must check if the item is the candidate (skip it) 
    # and if it fits in the remaining capacity.
    
    sorted_indices = np.argsort(-densities)
    
    best_candidate = -1
    best_total_value = -1
    
    for i in feasible_indices:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        
        cap_rem = remaining_capacity - w_i
        
        # Estimate value from remaining items
        # We iterate through the globally sorted list.
        # We skip item i.
        # We pick items if weight <= cap_rem.
        est_val = 0.0
        current_cap = cap_rem
        
        for j in sorted_indices:
            if j == i:
                continue
            w_j = remaining_weights[j]
            if w_j <= current_cap:
                est_val += remaining_values[j]
                current_cap -= w_j
            # We don't break immediately because a lighter item later might fit?
            # No, sorted by density doesn't mean sorted by weight.
            # A low density item might be very light and fit.
            # However, greedy by density is the standard heuristic.
            # Should we break if w_j > current_cap? 
            # No, because a subsequent item might be lighter.
            # But iterating all N items for each candidate is O(N^2).
            # This is acceptable for N=100.
            
        total_val = v_i + est_val
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_candidate = i
            
    return int(best_candidate)
