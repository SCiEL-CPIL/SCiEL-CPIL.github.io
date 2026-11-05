import numpy as np
import time

# Global deadline guard
_start_time = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin for 120s limit

def _dp_solve_and_pick(remaining_capacity, remaining_weights, remaining_values):
    """
    Solves the 0/1 knapsack exactly using DP and returns the index
    of one item from an optimal solution.
    Returns -1 if no item should be picked.
    Returns -2 if DP is not feasible (too large), signaling fallback.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Scale to integers
    scale = 1000
    C_int = int(round(remaining_capacity * scale))
    
    # If capacity is too large for DP, fall back to greedy
    if C_int > 200000:
        return -2
    
    w_ints = np.array([int(round(w * scale)) for w in remaining_weights])
    values = np.asarray(remaining_values, dtype=float)
    
    # Initialize DP table
    # dp[i][j] = max value using first i items with capacity j
    dp = np.zeros((n + 1, C_int + 1), dtype=float)
    
    # Fill DP table
    for i in range(1, n + 1):
        w = w_ints[i - 1]
        v = values[i - 1]
        
        if w > 0:
            # Copy previous row for capacities < w
            # If w > C_int, this copies the whole row
            dp[i, :min(w, C_int + 1)] = dp[i - 1, :min(w, C_int + 1)]
            
            if w <= C_int:
                # For capacities >= w
                dp[i, w:] = np.maximum(dp[i - 1, w:], dp[i - 1, :-w] + v)
            # else: item too heavy, dp[i, w:] is already copied from dp[i-1]
        else:
            # Weight 0
            if v > 0:
                dp[i, :] = dp[i - 1, :] + v
            else:
                dp[i, :] = dp[i - 1, :]

    # Backtrack to find one item in the optimal solution
    j = C_int
    for i in range(n, 0, -1):
        if dp[i, j] != dp[i - 1, j]:
            # Item i-1 is in the optimal solution
            return i - 1
        # If not taken, j remains same
        # We continue to i-1 with same j
        
    return -1

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Time check
    if time.time() - _start_time > TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    # Try DP first
    idx = _dp_solve_and_pick(remaining_capacity, remaining_weights, remaining_values)
    
    if idx != -2:
        return int(idx)
        
    # Fallback to Greedy if DP was too large
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    # Simple greedy: pick best density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights[feasible_indices] > 0, values[feasible_indices] / weights[feasible_indices], np.inf)
        
    best_local_idx = np.argmax(densities)
    return int(feasible_indices[best_local_idx])
