import numpy as np
import time

# Global timer to enforce the 150s limit across all calls
_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s kill

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using exact Dynamic Programming.
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
    
    # Extract feasible items
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    n_feas = len(w_feas)
    
    if n_feas == 0:
        return -1
        
    # Determine scale for integer weights
    # Capacity is typically 12.5. Weights are floats.
    # We want cap_int to be manageable for DP.
    # If weights have 2 decimal places, scale=100 is enough.
    # Let's try to detect precision or just use a robust scale.
    # Since C is fixed at 12.5 in the task description, and weights are likely generated with limited precision,
    # scale=1000 gives cap_int=12500.
    # DP size: 100 * 12501 = 1.25M floats. ~10MB. This is fast in numpy.
    
    scale = 1000
    cap_int = int(round(C * scale))
    
    if cap_int < 1:
        # If capacity is very small, just pick the lightest feasible item
        lightest_idx = np.argmin(w_feas)
        return int(feasible_indices[lightest_idx])

    w_int = np.round(w_feas * scale).astype(np.int64)
    
    # Ensure no negative weights (shouldn't happen, but safe)
    w_int = np.maximum(w_int, 0)
    
    # Check for zero weight items with positive value
    # If any w_int == 0 and v > 0, we should definitely take one.
    # Actually, DP handles this, but taking a zero-weight item doesn't reduce capacity.
    # If we have multiple zero-weight items, we take all of them.
    # The backtrack will pick the last one found.
    
    # DP Table: dp[i][c] = max value using first i items with capacity c
    # We use a 2D array for backtracking.
    # To optimize memory and speed, we can use a list of arrays or a single 2D array.
    # 2D array is fine.
    
    dp = np.zeros((n_feas + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n_feas + 1):
        wi = w_int[i-1]
        vi = v_feas[i-1]
        
        # Copy previous row
        dp[i, :] = dp[i-1, :]
        
        if wi <= cap_int:
            # Update: dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi) for c >= wi
            # Vectorized update
            if wi > 0:
                take_vals = dp[i-1, :cap_int+1-wi] + vi
                prev_vals = dp[i-1, wi:cap_int+1]
                dp[i, wi:cap_int+1] = np.maximum(prev_vals, take_vals)
            else:
                # wi == 0
                # dp[i, c] = max(dp[i-1, c], dp[i-1, c] + vi) = dp[i-1, c] + vi if vi > 0
                if vi > 0:
                    dp[i, :] += vi
                # If vi <= 0, we don't take it, so dp[i, :] remains dp[i-1, :]
    
    # Backtrack to find one item in the optimal solution
    c = cap_int
    selected_idx = -1
    
    for i in range(n_feas, 0, -1):
        wi = w_int[i-1]
        vi = v_feas[i-1]
        
        # Check if item i-1 was taken
        # If dp[i, c] > dp[i-1, c], it was taken.
        # Use a small epsilon for float comparison.
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected_idx = i - 1
            c -= wi
            break
        
    if selected_idx != -1:
        # Map back to original index
        return int(feasible_indices[selected_idx])
    else:
        # If no item was taken in the backtrack (shouldn't happen if optimal value > 0),
        # fallback to best density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v_feas / w_feas
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible_indices[best_local])
