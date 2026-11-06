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

def _dp_solve(cap, weights, values, scale=1.0):
    """
    Solve 0/1 knapsack using DP on scaled values.
    Returns (total_value, first_item_index) where first_item_index is the index
    of the first item in the optimal solution (or -1 if no items fit).
    """
    n = len(weights)
    if n == 0 or cap <= 0:
        return 0.0, -1
    
    # Scale values to integers
    scaled_values = np.round(values * scale).astype(np.int64)
    
    # Filter feasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return 0.0, -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_scaled_values = scaled_values[feasible_mask]
    n_feasible = len(feasible_indices)
    
    max_scaled_value = int(np.max(feasible_scaled_values)) if n_feasible > 0 else 0
    
    # DP table: dp[v] = minimum weight to achieve exactly v scaled value
    # Initialize with infinity
    INF = float('inf')
    dp = np.full(max_scaled_value + 1, INF, dtype=np.float64)
    dp[0] = 0.0
    
    # Track which item was used to achieve each state for backtracking
    # prev_item[v] = index of item that achieved value v
    prev_item = np.full(max_scaled_value + 1, -1, dtype=np.int64)
    prev_value = np.full(max_scaled_value + 1, 0, dtype=np.int64)
    
    for i in range(n_feasible):
        w_i = feasible_weights[i]
        v_i = int(feasible_scaled_values[i])
        if v_i <= 0:
            continue
        
        # Process in reverse to avoid using same item twice
        for v in range(max_scaled_value, v_i - 1, -1):
            if dp[v - v_i] + w_i < dp[v]:
                dp[v] = dp[v - v_i] + w_i
                prev_item[v] = i
                prev_value[v] = v - v_i
    
    # Find the maximum scaled value that fits within capacity
    best_scaled_value = 0
    for v in range(max_scaled_value, -1, -1):
        if dp[v] <= cap + 1e-9:
            best_scaled_value = v
            break
    
    if best_scaled_value == 0:
        return 0.0, -1
    
    # Backtrack to find the first item added
    # We need to trace back from best_scaled_value to find which item was added last
    # Then continue tracing to find the first item
    v = best_scaled_value
    first_item = -1
    last_item = -1
    
    while v > 0 and prev_item[v] != -1:
        item_idx = prev_item[v]
        if first_item == -1:
            first_item = item_idx
        last_item = item_idx
        v = prev_value[v]
    
    if first_item == -1:
        # No valid path found, fallback to greedy
        return 0.0, -1
    
    total_value = float(feasible_scaled_values[first_item]) / scale
    
    return total_value, int(feasible_indices[first_item])

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
    
    # Use DP with value scaling to find near-optimal first item
    # Try different scales to balance accuracy and speed
    
    best_total_value = -1
    best_first_index = feasible_indices[0]
    
    # Try scale = 1.0 first (exact if values are integers)
    for scale in [1.0, 2.0, 5.0]:
        if not _check_time():
            break
        
        total_val, first_idx = _dp_solve(cap, feasible_weights, feasible_values, scale)
        
        if first_idx != -1 and total_val > best_total_value:
            best_total_value = total_val
            best_first_index = first_idx
    
    # If DP didn't find a good solution, fall back to greedy strategies
    if best_total_value <= 0:
        # Compute density order
        density = feasible_values / feasible_weights
        order_density = np.argsort(-density)
        
        # Simulate pure density strategy
        density_total = _greedy_fill(cap, feasible_weights, feasible_values, order_density)
        density_first = feasible_indices[order_density[0]]
        
        # Compute value order
        order_value = np.argsort(-feasible_values)
        value_total = _greedy_fill(cap, feasible_weights, feasible_values, order_value)
        value_first = feasible_indices[order_value[0]]
        
        if density_total > value_total:
            best_total_value = density_total
            best_first_index = density_first
        else:
            best_total_value = value_total
            best_first_index = value_first
    
    return int(best_first_index)
# EVOLVE-BLOCK-END
