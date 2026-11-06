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
    # Safety margin: return True if we have more than 10 seconds left
    return (time.time() - _START_TIME) < 135.0

def _greedy_fill_value(weights, values, order, cap):
    """
    Simulate greedy fill following a given order of indices.
    Returns total value achievable.
    Uses numpy arrays for speed.
    """
    if cap <= 0:
        return 0.0
    total_val = 0.0
    cap_rem = cap
    for idx in order:
        if cap_rem <= 0:
            break
        w = weights[idx]
        if w <= cap_rem:
            cap_rem -= w
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
    
    # Compute density for sorting
    density = feasible_values / feasible_weights
    density_order = np.argsort(-density, kind='mergesort')
    
    # Precompute sorted arrays by density for fast greedy simulation
    sorted_weights = feasible_weights[density_order]
    sorted_values = feasible_values[density_order]
    
    best_total_value = -1
    best_first_index = feasible_indices[0]
    
    # For each candidate first item, simulate: pick it, then greedily fill rest by density
    # To speed this up, we precompute the density-sorted order and skip the chosen item
    for i in range(n_feasible):
        if not _check_time():
            break
        
        w_i = feasible_weights[i]
        v_i = feasible_values[i]
        remaining_cap = cap - w_i
        
        if remaining_cap <= 0:
            total_val = v_i
        else:
            total_val = v_i
            cap_rem = remaining_cap
            # Iterate through density-sorted items, skipping item i
            for pos in range(n_feasible):
                if cap_rem <= 0:
                    break
                orig_idx = density_order[pos]
                if orig_idx == i:
                    continue
                w_j = feasible_weights[orig_idx]
                if w_j <= cap_rem:
                    cap_rem -= w_j
                    total_val += feasible_values[orig_idx]
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_first_index = feasible_indices[i]
    
    # Also evaluate a few alternative strategies
    # Strategy 1: Pure density (first item in density order)
    if n_feasible > 0:
        density_first_idx = density_order[0]
        density_first_global = feasible_indices[density_first_idx]
        density_total = _greedy_fill_value(feasible_weights, feasible_values, density_order, cap)
        if density_total > best_total_value:
            best_total_value = density_total
            best_first_index = density_first_global
    
    # Strategy 2: Pure value (highest value first)
    if n_feasible > 0:
        value_order = np.argsort(-feasible_values, kind='mergesort')
        value_first_idx = value_order[0]
        value_first_global = feasible_indices[value_first_idx]
        value_total = _greedy_fill_value(feasible_weights, feasible_values, value_order, cap)
        if value_total > best_total_value:
            best_total_value = value_total
            best_first_index = value_first_global
    
    # Strategy 3: Weight ascending (lightest first)
    if n_feasible > 0:
        weight_order = np.argsort(feasible_weights, kind='mergesort')
        weight_first_idx = weight_order[0]
        weight_first_global = feasible_indices[weight_first_idx]
        weight_total = _greedy_fill_value(feasible_weights, feasible_values, weight_order, cap)
        if weight_total > best_total_value:
            best_total_value = weight_total
            best_first_index = weight_first_global
    
    return int(best_first_index)
# EVOLVE-BLOCK-END
