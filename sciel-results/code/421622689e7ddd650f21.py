"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: select the next item to add.
    
    Strategy: Exact Dynamic Programming with weight scaling.
    We scale the weights to integers (factor 1000) to allow for an exact
    0/1 knapsack solution on the remaining items. This ensures that the
    item selected is part of the optimal solution for the remaining subproblem.
    """
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    capacity = float(remaining_capacity)
    
    # Check if any items are feasible
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    n = len(feasible_weights)
    
    # Scale weights and capacity to integers
    # Using a scale factor of 1000 to balance precision and performance
    scale = 1000
    scaled_weights = np.round(feasible_weights * scale).astype(np.int64)
    scaled_capacity = int(np.round(capacity * scale))
    
    # Filter out items with 0 scaled weight (edge case) or if capacity is 0
    if scaled_capacity <= 0:
        return -1
        
    # Ensure scaled weights are at least 1 if original weight > 0
    # If a weight is very small but non-zero, it might round to 0.
    # We should handle this. If w > 0 but scaled_w == 0, treat it as 1?
    # Or just keep it as is. If scaled_w == 0, it can be taken for free.
    # Let's keep scaled_w as is.
    
    # DP Initialization
    # dp[c] = max value achievable with capacity c
    dp = np.zeros(scaled_capacity + 1, dtype=np.float64)
    
    # take[i][c] = 1 if item i is taken in the optimal solution for capacity c
    # We use a 2D array for backtracking.
    # n items, scaled_capacity + 1 capacities
    # To save memory and time, we can store the choice in a list of arrays or a single 2D array.
    # n is max 100, C is max 12500. 100 * 12500 = 1.25M bytes (if uint8).
    take = np.zeros((n, scaled_capacity + 1), dtype=np.uint8)
    
    # Fill DP table
    for i in range(n):
        w = scaled_weights[i]
        v = feasible_values[i]
        
        # If weight is 0, we can always take it (if value > 0)
        if w == 0:
            if v > 0:
                take[i] = 1
                dp += v # This is not quite right for 0/1, but if w=0, it's independent
                # Actually, if w=0, dp[c] should be updated for all c.
                # But standard 0/1 knapsack with w=0:
                # dp_new[c] = max(dp_old[c], dp_old[c] + v) = dp_old[c] + v (if v>0)
                # So we can just add v to all dp entries?
                # Wait, if we have multiple w=0 items, we take all of them.
                # Let's handle w=0 separately to avoid breaking the loop logic.
                pass 
        
        # Standard 0/1 Knapsack update
        # Iterate backwards to avoid using the same item twice
        if w > 0:
            # We need to update dp and take[i]
            # For c from scaled_capacity down to w:
            #   if dp[c-w] + v > dp[c]:
            #       dp[c] = dp[c-w] + v
            #       take[i][c] = 1
            #   else:
            #       take[i][c] = 0
            
            # Vectorized approach is hard for the dependency, but we can try to optimize the loop.
            # Python loop is slow. Let's see if we can use a trick.
            # No, we must loop.
            
            # Optimization: Only loop down to w.
            # dp[w:scaled_capacity+1] depends on dp[0:scaled_capacity+1-w]
            
            # Let's use a local variable for dp to speed up access
            dp_local = dp
            take_row = take[i]
            
            for c in range(scaled_capacity, w - 1, -1):
                val = dp_local[c - w] + v
                if val > dp_local[c]:
                    dp_local[c] = val
                    take_row[c] = 1
                else:
                    take_row[c] = 0
        else:
            # w == 0
            if v > 0:
                take[i] = 1
                dp += v

    # Backtrack to find the set of items included in the optimal solution
    # Start from the last item and full capacity
    included_indices = []
    c = scaled_capacity
    for i in range(n - 1, -1, -1):
        if take[i][c] == 1:
            included_indices.append(i)
            c -= scaled_weights[i]
    
    if not included_indices:
        return -1
        
    # Select the best item from the optimal set to return
    # Strategy: Return the item with the highest density (value/weight)
    # This is a heuristic choice within the optimal set.
    # It might help in case of multiple optimal sets or to guide the next steps.
    
    best_local_idx = included_indices[0]
    best_density = -1.0
    
    for idx in included_indices:
        w = feasible_weights[idx]
        v = feasible_values[idx]
        if w > 0:
            density = v / w
        else:
            density = float('inf') if v > 0 else 0.0
            
        if density > best_density:
            best_density = density
            best_local_idx = idx
            
    best_global_idx = int(feasible_indices[best_local_idx])
    return best_global_idx
# EVOLVE-BLOCK-END
