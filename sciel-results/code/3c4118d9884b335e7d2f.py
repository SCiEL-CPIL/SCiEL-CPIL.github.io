import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Convert to numpy arrays for speed
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Filter out infeasible items (weight > capacity) and zero/negative weight items
    # If weight == 0 and value > 0, we should definitely pick it.
    # If weight < 0, it's a "profit" item, always pick.
    # Standard knapsack assumes weight > 0.
    
    # Handle zero weight, positive value items immediately
    zero_w_pos_v_mask = (weights == 0) & (values > 0)
    if np.any(zero_w_pos_v_mask):
        return int(np.where(zero_w_pos_v_mask)[0][0])
    
    # Handle negative weight, positive value items (always beneficial)
    neg_w_pos_v_mask = (weights < 0) & (values > 0)
    if np.any(neg_w_pos_v_mask):
        return int(np.where(neg_w_pos_v_mask)[0][0])
    
    # Filter feasible items
    feasible_mask = (weights > 0) & (weights <= remaining_capacity)
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
        
    # Prepare items for B&B
    # Sort by density (value/weight) descending
    densities = f_values / f_weights
    sorted_order = np.argsort(-densities)
    
    s_weights = f_weights[sorted_order]
    s_values = f_values[sorted_order]
    s_indices = feasible_indices[sorted_order] # Map back to original indices
    
    # B&B Solver
    # State: index, current_value, current_weight
    # We want to find the set of items that maximizes value.
    # Then pick one from that set.
    
    best_value = -1
    best_set_indices = [] # Stores original indices of the best solution found
    
    start_time = time.time()
    time_limit = 1.0 # 1 second per call is generous, but safe.
    
    # Precompute suffix sums for fractional bound? 
    # Or just compute on the fly. On the fly is O(N) per node.
    # With pruning, number of nodes is small.
    
    def fractional_bound(idx, curr_val, curr_weight):
        # Calculate upper bound using fractional knapsack from idx onwards
        ub = curr_val
        rem_cap = remaining_capacity - curr_weight
        if rem_cap <= 0:
            return ub
            
        for j in range(idx, f_n):
            if s_weights[j] <= rem_cap:
                ub += s_values[j]
                rem_cap -= s_weights[j]
            else:
                if s_weights[j] > 0:
                    ub += s_values[j] * (rem_cap / s_weights[j])
                break
        return ub

    def dfs(idx, curr_val, curr_weight, curr_set):
        nonlocal best_value, best_set_indices
        
        if time.time() - start_time > time_limit:
            return
            
        if idx == f_n:
            if curr_val > best_value:
                best_value = curr_val
                best_set_indices = list(curr_set)
            return
            
        # Pruning
        # Calculate bound
        # Optimization: If curr_val + sum of remaining values < best_value, prune
        # But fractional bound is tighter.
        
        # Quick check: if curr_val + sum(s_values[idx:]) <= best_value, prune
        # Precompute suffix sums of values?
        
        ub = fractional_bound(idx, curr_val, curr_weight)
        if ub <= best_value:
            return
            
        # Include item idx
        w = s_weights[idx]
        v = s_values[idx]
        
        # Try Include first (heuristic: good solutions often include high density items)
        if curr_weight + w <= remaining_capacity:
            curr_set.append(s_indices[idx])
            dfs(idx + 1, curr_val + v, curr_weight + w, curr_set)
            curr_set.pop()
            
        # Try Exclude
        dfs(idx + 1, curr_val, curr_weight, curr_set)
        
    curr_set = []
    dfs(0, 0.0, 0.0, curr_set)
    
    # If B&B timed out or found nothing, fallback to greedy
    if not best_set_indices:
        # Greedy: pick best density item that fits
        # We already sorted by density.
        # Find first item in sorted order that fits?
        # Or just pick the first feasible one?
        # Let's pick the one with highest value/weight ratio that fits.
        # Since s_weights/s_values are sorted by density, the first one is best density.
        # But we need to check if it fits.
        # Actually, if B&B failed, we just need *a* valid item.
        # Let's return the first feasible item (highest density).
        return int(s_indices[0])
        
    # Pick one item from the optimal set
    # Any item in the optimal set is a valid choice for the next step
    # because the subproblem (remaining items, remaining capacity) 
    # will still allow reaching the optimal total value?
    # Yes, if S* is the optimal set, removing any i in S* leaves S* \ {i} 
    # as a feasible solution for the subproblem, and the optimal value of the subproblem 
    # is at least value(S* \ {i}). 
    # Actually, we want to ensure we don't lose optimality.
    # The standard property of 0/1 knapsack is that if you pick an item from the optimal set,
    # the remaining optimal solution for the rest of the items will complete it.
    
    # Pick the first one in the set
    return int(best_set_indices[0])
