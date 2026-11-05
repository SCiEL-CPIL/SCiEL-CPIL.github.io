"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)

    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    # Approach: Dynamic Programming on scaled capacity
    # This is a different approach family (Exact/DP vs Heuristic/Greedy Lookahead)
    
    # Scale weights to integers to allow DP
    # Capacity is small (12.5), so scaling by 1000 gives 12500 states.
    # N=100, so 100 * 12500 = 1.25M operations. Very fast.
    scale = 1000
    cap_int = int(remaining_capacity * scale)
    
    # Check if weights can be scaled to integers without significant loss
    w_float = weights
    w_scaled = w_float * scale
    w_int_float = np.round(w_scaled)
    
    # If weights are not close to integers after scaling, DP might be inaccurate
    # We check if the difference is small. If not, we fallback to a robust greedy.
    # For generated instances, weights are often "nice" numbers.
    if not np.allclose(w_scaled, w_int_float, atol=1e-6):
        # Fallback: Greedy by Density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(weights > 0, values / weights, np.inf)
        sorted_indices = np.argsort(-densities)
        for idx in sorted_indices:
            if weights[idx] <= remaining_capacity:
                return int(idx)
        return -1

    w_int = w_int_float.astype(int)
    
    # Ensure no negative weights or weights > capacity (handled by mask, but w_int might exceed cap_int)
    # DP Table: dp[i][c] = max value using items i..n-1 with capacity c
    # We use a 2D array for easy reconstruction
    # Size: (n+1) x (cap_int + 1)
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    
    # Fill DP table from bottom up
    for i in range(n - 1, -1, -1):
        w_i = w_int[i]
        v_i = values[i]
        
        if w_i > cap_int:
            dp[i] = dp[i + 1]
        else:
            # Skip case: dp[i+1][c]
            # Take case: v_i + dp[i+1][c - w_i]
            
            # Initialize take array with -inf
            take = np.full(cap_int + 1, -np.inf)
            
            # For c >= w_i, take is possible
            if w_i <= cap_int:
                take[w_i:] = v_i + dp[i + 1][:cap_int - w_i + 1]
            
            dp[i] = np.maximum(dp[i + 1], take)
    
    # Reconstruct the first item
    c = cap_int
    for i in range(n):
        w_i = w_int[i]
        if w_i <= c:
            # Check if item i is part of the optimal solution for state (i, c)
            # dp[i][c] should be equal to values[i] + dp[i+1][c - w_i]
            # Use isclose to handle floating point precision
            if np.isclose(values[i] + dp[i + 1][c - w_i], dp[i][c], rtol=1e-9, atol=1e-9):
                return int(i)
        # If item i is not taken, we implicitly "skip" it and move to i+1 with same capacity c
        # The loop continues to i+1
    
    # If no item found via DP (should not happen if feasible items exist and optimal > 0)
    # Fallback to Greedy by Density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, np.inf)
    sorted_indices = np.argsort(-densities)
    for idx in sorted_indices:
        if weights[idx] <= remaining_capacity:
            return int(idx)
    return -1
# EVOLVE-BLOCK-END
