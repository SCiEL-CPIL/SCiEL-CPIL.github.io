import numpy as np
import time

# Global timer to enforce the 120s limit across all calls
_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a DP-based lookahead.
    Returns the index into remaining_weights/remaining_values, or -1 to stop.
    """
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights, dtype=np.float64)
        v = np.array(remaining_values, dtype=np.float64)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # If the number of feasible items is small, we can afford higher precision or just pick best
    # But let's try DP for all.
    
    # Extract feasible items to reduce N in DP if many are infeasible.
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    n_feas = len(w_feas)
    
    if n_feas == 0:
        return -1
    
    # Choose scale based on problem size to balance precision and speed
    # Target: DP table should be <= ~5M entries for safety
    # n_feas * cap_int <= 5,000,000
    # cap_int = C * scale
    # So scale <= 5,000,000 / (n_feas * C)
    max_scale = 5000000.0 / (n_feas * C) if C > 0 else 100.0
    # Also cap at reasonable values
    max_scale = min(max_scale, 1000.0)
    # Use a power-of-10 scale for clean rounding
    if max_scale >= 1000:
        scale = 1000
    elif max_scale >= 100:
        scale = 100
    elif max_scale >= 10:
        scale = 10
    else:
        scale = 1
    
    cap_int = int(round(C * scale))
    if cap_int < 1:
        # If capacity is very small, just pick the lightest feasible item
        lightest_idx = np.argmin(w_feas)
        return int(feasible_indices[lightest_idx])

    w_int = np.round(w_feas * scale).astype(np.int64)
    
    # Ensure no negative weights
    w_int = np.maximum(w_int, 0)
    
    # DP
    # dp[i] is an array of length cap_int+1
    # We need the full 2D table for backtracking.
    # dp = np.zeros((n_feas + 1, cap_int + 1), dtype=np.float64)
    
    # Optimization: Use 1D DP but store the "take" decisions?
    # Or just use 2D. It's small enough.
    
    dp = np.zeros((n_feas + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n_feas + 1):
        wi = w_int[i-1]
        vi = v_feas[i-1]
        
        # Copy previous row
        dp[i, :] = dp[i-1, :]
        
        if wi <= cap_int:
            # Update
            # dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
            # Vectorized
            take_vals = dp[i-1, :cap_int+1-wi] + vi
            prev_vals = dp[i-1, wi:cap_int+1]
            
            # We want to update dp[i, wi:cap_int+1]
            # max(prev_vals, take_vals)
            # Note: take_vals has length cap_int+1-wi
            # prev_vals has length cap_int+1-wi
            
            updated = np.maximum(prev_vals, take_vals)
            dp[i, wi:cap_int+1] = updated
        # else: item too heavy, row is already copied
    
    # Find all items that are part of at least one optimal solution
    # An item i is "definitely in an optimal solution" if dp[i][c] > dp[i-1][c]
    # But we want items that are in SOME optimal solution.
    # Item i is in some optimal solution if:
    #   dp[i][c] == dp[i-1][c] + vi AND dp[i-1][c-wi] + vi == dp[i][c]
    #   OR dp[i][c] > dp[i-1][c] (forced take)
    # More precisely: item i is in some optimal solution for capacity c if
    #   dp[i][c] == dp[i-1][c-wi] + vi (taking it achieves optimal)
    #   AND dp[i-1][c-wi] is achievable (which it always is in our DP)
    # But we also need that NOT taking it doesn't give a better result:
    #   dp[i-1][c] <= dp[i][c] (always true by DP definition)
    # So: item i is in some optimal solution iff dp[i-1][c-wi] + vi == dp[i][c]
    
    c = cap_int
    optimal_candidates = []
    
    for i in range(1, n_feas + 1):
        wi = w_int[i-1]
        vi = v_feas[i-1]
        
        if wi <= c:
            # Check if taking item i-1 with remaining capacity c-wi achieves optimal
            if abs(dp[i-1, c-wi] + vi - dp[i, c]) < 1e-9:
                # This item is part of at least one optimal solution
                # Score it by: prefer lighter items (more remaining capacity)
                # and higher value (better immediate gain)
                # Use a combined score: value / (weight + epsilon) but prefer light
                # Actually, for future flexibility, lighter is better.
                # Let's score by: -weight (lighter is better) as primary, value as secondary
                score = -w_feas[i-1]  # lighter is better
                optimal_candidates.append((score, i-1))
    
    if optimal_candidates:
        # Pick the candidate with the best score (lightest weight)
        # If tie, pick highest value
        optimal_candidates.sort(key=lambda x: (x[0], -v_feas[x[1]]), reverse=True)
        selected_idx = optimal_candidates[0][1]
        return int(feasible_indices[selected_idx])
    else:
        # Fallback: no item is in an optimal solution (shouldn't happen), use density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v_feas / w_feas
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible_indices[best_local])
