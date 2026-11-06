"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _check_time():
    if _START_TIME is None:
        return True
    # Safety margin: return True if we have more than 5 seconds left
    return (time.time() - _START_TIME) < 140.0

def _greedy_fill(cap, weights, values, order):
    """
    Simulate greedy fill following a given order of indices.
    Returns total value achievable.
    """
    total_val = 0.0
    cap_remaining = cap
    for idx in order:
        if cap_remaining <= 0:
            break
        w = weights[idx]
        if w <= cap_remaining:
            cap_remaining -= w
            total_val += values[idx]
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = remaining_capacity
    
    # Filter out infeasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    n_feasible = len(feasible_indices)
    
    # If only one feasible item, take it
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # Compute various sorting orders
    # 1. Value density (value/weight) descending
    density = feasible_values / feasible_weights
    order_density = np.argsort(-density)
    
    # 2. Value descending
    order_value = np.argsort(-feasible_values)
    
    # 3. Weight ascending (lightest first)
    order_weight = np.argsort(feasible_weights)
    
    # 4. Value/weight^2 (heavier penalty)
    order_density2 = np.argsort(-(feasible_values / (feasible_weights ** 2)))
    
    # 5. Value * log(1 + 1/weight) - hybrid
    hybrid_score = feasible_values * np.log1p(1.0 / feasible_weights)
    order_hybrid = np.argsort(-hybrid_score)
    
    # Evaluate each strategy by simulating the full greedy fill
    # For each candidate first item, we check the total value if we pick it first
    # then continue with the remaining items in the strategy's order
    
    # We'll evaluate the top K candidates from each strategy
    # But to be efficient, let's just evaluate the full strategy for each sorting order
    # and pick the first item from the best strategy
    
    # Actually, a better approach: for each feasible item, compute the total value
    # if we pick it first, then greedily fill the rest with the best density order
    # This is O(n^2) but n is at most 100, so it's fine
    
    # Precompute the density-sorted order for the remaining items
    # For each candidate i, we need to simulate: pick i, then greedily fill rest by density
    
    # Let's do a smarter approach:
    # For each candidate first item i:
    #   remaining_cap = cap - w_i
    #   remaining_items = all feasible items except i
    #   greedily fill remaining by density order
    #   total_value = v_i + greedy_fill_value
    
    # To make this efficient, precompute prefix sums or use a fast simulation
    
    best_total_value = -1
    best_first_index = feasible_indices[0]
    
    # Sort feasible items by density for the greedy fill part
    density_order = np.argsort(-density)
    
    # For each candidate, simulate
    for i in range(n_feasible):
        if not _check_time():
            break
        
        w_i = feasible_weights[i]
        v_i = feasible_values[i]
        remaining_cap = cap - w_i
        
        if remaining_cap <= 0:
            total_val = v_i
        else:
            # Greedily fill remaining items by density, excluding item i
            total_val = v_i
            cap_rem = remaining_cap
            for j in density_order:
                if j == i:
                    continue
                w_j = feasible_weights[j]
                if w_j <= cap_rem:
                    cap_rem -= w_j
                    total_val += feasible_values[j]
                if cap_rem <= 0:
                    break
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_first_index = feasible_indices[i]
    
    # Also try other strategies as fallbacks if the above didn't find something good
    # Let's also evaluate pure density strategy's first item
    density_first = feasible_indices[density_order[0]]
    # And pure value strategy's first item  
    value_first = feasible_indices[order_value[0]]
    
    # Simulate pure density strategy
    density_total = _greedy_fill(cap, feasible_weights, feasible_values, density_order)
    # Simulate pure value strategy
    value_total = _greedy_fill(cap, feasible_weights, feasible_values, order_value)
    
    # If pure strategies are better, use their first item
    if density_total > best_total_value:
        best_total_value = density_total
        best_first_index = density_first
    if value_total > best_total_value:
        best_total_value = value_total
        best_first_index = value_first
    
    return int(best_first_index)
# EVOLVE-BLOCK-END
