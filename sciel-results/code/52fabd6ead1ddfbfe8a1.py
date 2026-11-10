import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)

    # Handle zero weight, positive value items immediately
    zero_w_pos_v_mask = (weights == 0) & (values > 0)
    if np.any(zero_w_pos_v_mask):
        return int(np.where(zero_w_pos_v_mask)[0][0])

    # Handle negative weight, positive value items (always beneficial)
    neg_w_pos_v_mask = (weights < 0) & (values > 0)
    if np.any(neg_w_pos_v_mask):
        return int(np.where(neg_w_pos_v_mask)[0][0])

    # Filter feasible items (weight > 0 and weight <= remaining_capacity)
    feasible_mask = (weights > 0) & (weights <= remaining_capacity)
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)

    if f_n == 0:
        return -1

    # Determine scaling factor for DP
    C = remaining_capacity
    # Target operations ~ 5 million to stay well within time limits per call
    # We have 100 items, C=12.5. 
    # If scale=1000, C_int=12500. 100 * 12500 = 1.25M ops. Very fast.
    # If scale=10000, C_int=125000. 100 * 125000 = 12.5M ops. ~0.1-0.2s in NumPy? Maybe slower.
    # Let's aim for C_int around 5000-10000 for safety and speed.
    target_c_int = 5000
    scale = target_c_int / C if C > 0 else 1
    scale = int(scale)
    if scale < 1:
        scale = 1

    C_int = int(C * scale)
    w_ints = (f_weights * scale).astype(np.int64)
    # Ensure weights are at least 1 if they are positive
    w_ints[w_ints == 0] = 1

    # Check if DP is feasible
    if C_int > 50000 or f_n * C_int > 10_000_000:
        # Fall back to greedy density
        densities = f_values / f_weights
        best_idx = np.argmax(densities)
        return int(feasible_indices[best_idx])

    # DP for 0/1 Knapsack
    # dp[j] = max value achievable with capacity j
    dp = np.zeros(C_int + 1, dtype=float)
    # takes[i][j] = True if item i is taken in optimal solution for capacity j
    # To save memory and time, we can just backtrack using a 2D boolean array or re-compute.
    # With f_n=100 and C_int=5000, 100*5000 = 500k booleans. Very small.
    takes = np.zeros((f_n, C_int + 1), dtype=bool)

    for i in range(f_n):
        w = w_ints[i]
        v = f_values[i]
        if w > C_int:
            continue
        # Update dp from right to left to ensure 0/1 constraint
        # We need to update dp[j] for j from C_int down to w
        # dp[j] = max(dp[j], dp[j-w] + v)
        
        # Vectorized update is tricky for 0/1 because of the dependency.
        # However, we can do:
        # candidate = dp[:-w] + v
        # But this uses the OLD dp values?
        # If we do dp[w:] = np.maximum(dp[w:], dp[:-w] + v), 
        # numpy evaluates RHS first. dp[:-w] is a view of the current dp.
        # So it uses the values from the previous iteration (or earlier in this iteration if we were doing 1/0, but we are doing 0/1).
        # Wait, in 0/1 knapsack, we iterate items. For a fixed item i, we update dp[j] based on dp[j-w] from the state BEFORE item i was considered.
        # Since we are updating dp in place, we must iterate j from C_int down to w.
        # Vectorization with in-place update from right to left is not directly supported by simple slicing because dp[w:] and dp[:-w] overlap?
        # No, dp[w:] is indices w..C_int. dp[:-w] is indices 0..C_int-w.
        # They do not overlap in a way that causes issues for reading if we read all before writing?
        # `np.maximum(dp[w:], dp[:-w] + v)` reads `dp[w:]` and `dp[:-w]`.
        # If `w` is small, `dp[w:]` and `dp[:-w]` overlap?
        # Example: w=1. dp[1:] and dp[:-1] (dp[0:-1]). They overlap.
        # If we write to dp[1:], does it affect dp[:-1] which is being read?
        # In `a = np.maximum(b, c)`, b and c are evaluated first. So `dp[w:]` is read into a temporary buffer?
        # Actually, `np.maximum` returns a new array. The inputs are read.
        # So `new_vals = np.maximum(dp[w:], dp[:-w] + v)` is safe.
        # Then `dp[w:] = new_vals`.
        
        # But we also need to record `takes`.
        # takes[i][j] is True if dp[j-w] + v > dp[j] (old).
        
        old_dp_w = dp[w:].copy()
        candidate = dp[:-w] + v
        mask = candidate > old_dp_w
        # Update dp
        dp[w:] = np.where(mask, candidate, old_dp_w)
        # Record takes
        takes[i][w:] = mask

    # Backtrack to find one item to select
    # Start at full capacity
    c = C_int
    selected_items = []
    for i in range(f_n - 1, -1, -1):
        if takes[i][c]:
            selected_items.append(i)
            c -= w_ints[i]
            if c < 0:
                break

    if not selected_items:
        # Fallback to greedy
        densities = f_values / f_weights
        best_idx = np.argmax(densities)
        return int(feasible_indices[best_idx])

    # Pick the first item in the optimal set (or any)
    # The order in selected_items is reverse of processing (last item considered first).
    # It doesn't matter which one we pick, as long as it's in the optimal set.
    # Picking the one with highest density among selected might be good for future steps?
    # Or just pick the first one found in backtrack (which is the last item in the 0..N-1 order).
    # Let's pick the one with the highest value/weight ratio among the selected items to guide the next steps?
    # Actually, the DP solution is optimal for the CURRENT state.
    # Removing any item from the optimal set leaves a sub-problem that can still be solved optimally by the same DP logic in the next call.
    # So we can just return the index of any selected item.
    
    # Let's return the index of the item that was "last" added in the DP sense (highest i in selected_items)?
    # selected_items[0] is the highest i.
    f_idx = selected_items[0]
    return int(feasible_indices[f_idx])
