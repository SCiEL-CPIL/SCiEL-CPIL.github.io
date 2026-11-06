"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack.
    
    Strategy: 
    For each feasible item i, estimate the total value if we pick i and then 
    fill the remaining capacity with the optimal fractional knapsack solution 
    of the remaining items. Pick the item i that maximizes this estimate.
    """
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    capacity = remaining_capacity
    
    n = len(weights)
    if n == 0 or capacity <= 0:
        return -1

    # Handle zero weights
    # If weight is 0, density is effectively infinity if value > 0
    # We handle this in sorting by putting them first if value > 0
    
    # Calculate densities for sorting
    # To avoid division by zero, we can use a large number for zero weights with positive value
    # Or simply sort by (value/weight) with special handling.
    
    # Vectorized sort is faster, but we need to iterate for the lookahead.
    # Let's get the global order by density.
    
    # Create a safe density array
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    
    # Items with weight 0 and value > 0 should be at the top.
    # Items with weight 0 and value 0 can be anywhere.
    # Items with weight > 0 use calculated density.
    
    # For sorting:
    # We want descending density.
    # np.nan_to_num can help if we get inf/nan, but let's be precise.
    
    # Sort indices by density descending.
    # We need to handle the case where weights are 0.
    # If w=0, v>0 -> inf. If w=0, v=0 -> 0 (or -inf, doesn't matter much as v=0).
    
    # Let's use a stable sort on a tuple or a modified density.
    # Simpler: just use argsort on densities, replacing inf with a large number.
    # Actually, np.argsort handles inf correctly (puts them first for descending? No, inf is largest, so for descending (reverse=True) they are first).
    
    # But we must ensure we don't pick an item with w > capacity.
    
    # 1. Identify feasible items
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return -1
        
    # 2. Get global order of all items by density
    # We need to sort all items, but we will skip infeasible ones or the candidate i in the inner loop?
    # Actually, the fractional bound calculation should only consider items that fit in the *remaining* capacity 
    # of the fractional problem. The fractional algorithm naturally handles this by stopping when capacity is hit.
    # However, items with w > remaining_capacity (in the sub-problem) will just be skipped or cause the fractional part to be 0 if they are the only ones?
    # No, the fractional algorithm sorts by density and fills. If an item doesn't fit, it takes a fraction.
    # So we just need the global density order.
    
    # Replace nan with 0 (for w=0, v=0 case? 0/0 is nan). 
    # If w=0, v=0, density is 0 effectively.
    # If w=0, v>0, density is inf.
    
    # Let's create a sort key.
    # We want items with w=0, v>0 first.
    # Then items with w>0 by v/w descending.
    # Then items with w=0, v=0.
    
    # Simple approach:
    # sort_key = densities
    # But we need to handle inf and nan.
    
    # Let's just use np.argsort on densities with a trick.
    # np.nan_to_num(densities, nan=0.0, posinf=1e18)
    # Then sort descending.
    
    safe_densities = np.nan_to_num(densities, nan=0.0, posinf=1e18, neginf=0.0)
    global_order = np.argsort(safe_densities)[::-1]
    
    best_score = -1
    best_index = -1
    
    # Iterate through feasible items as candidates
    # Optimization: We can iterate through all items, but skip infeasible ones.
    
    for i in range(n):
        if weights[i] > capacity:
            continue
            
        cap_rem = capacity - weights[i]
        val_frac = 0.0
        
        # Traverse global order to compute fractional bound for items != i
        # Note: We must skip item i.
        # Also, if an item j in the order has weight > cap_rem, we take a fraction.
        
        for j in global_order:
            if j == i:
                continue
                
            w_j = weights[j]
            v_j = values[j]
            
            if w_j == 0:
                # If weight is 0, we can take all of it if it fits (it always fits)
                # But wait, if we take it, it doesn't reduce capacity.
                # In the fractional knapsack, items with 0 weight are taken fully if value > 0.
                if v_j > 0:
                    val_frac += v_j
                # If v_j == 0, it adds nothing.
                # Note: If w_j == 0, it doesn't consume capacity, so we don't break.
                # We continue to next item.
            else:
                if w_j <= cap_rem:
                    cap_rem -= w_j
                    val_frac += v_j
                else:
                    # Take fraction
                    if cap_rem > 0:
                        val_frac += v_j * (cap_rem / w_j)
                    # cap_rem becomes 0
                    # We break because no more capacity
                    break
            
            # If cap_rem becomes 0, we can break early?
            # Yes, if cap_rem == 0, no more value can be added.
            if cap_rem == 0:
                break
                
        total_score = values[i] + val_frac
        
        if total_score > best_score:
            best_score = total_score
            best_index = i
            
    return best_index
# EVOLVE-BLOCK-END
