"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 135.0  # Safety margin before 150s limit
_SCALE = 1000

def _solve_dp(w, v, C, scale):
    """
    Solves 0/1 knapsack exactly using DP on scaled integer weights.
    Returns (list of indices in optimal solution, total value).
    """
    n = len(w)
    if n == 0 or C <= 0:
        return [], 0.0

    cap_int = int(round(C * scale))
    if cap_int < 0:
        return [], 0.0

    # dp[i][c] = max value using first i items with capacity c
    # Using numpy for vectorized row updates
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)

    for i in range(1, n + 1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]

        if wi <= cap_int:
            prev_row = dp[i-1, :]
            # Copy previous row as base (not taking item)
            dp[i, :] = prev_row
            # Update for taking item: dp[i, wi:] = max(dp[i-1, wi:], dp[i-1, :cap_int-wi] + vi)
            take_val = prev_row[:cap_int - wi + 1] + vi
            curr_row = dp[i, wi:]
            np.maximum(curr_row, take_val, out=curr_row)
        else:
            # Item too heavy, just copy previous row
            dp[i, :] = dp[i-1, :]

    # Backtrack to find the items in the optimal solution
    selected = []
    c = cap_int
    for i in range(n, 0, -1):
        # Check if item i-1 was taken
        # dp[i][c] > dp[i-1][c] means item was taken
        # Use tolerance for floating point comparison
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected.append(i - 1)
            wi = int(round(w[i-1] * scale))
            c -= wi
            if c < 0:
                c = 0
    
    selected.reverse()
    return selected, dp[n, cap_int]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    
    Strategy: Solve the full 0/1 knapsack DP for the current subproblem.
    Return any item from the optimal set.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check time
    if time.time() - _start_time > _TIMEOUT:
        return -1

    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # If capacity is very small or n is large, maybe fall back? 
    # But with 100 items and cap 12.5, DP is fast.
    # Scale 1000 -> cap_int ~ 12500.
    # DP size: 101 * 12500 ~ 1.26M floats. Very fast.
    
    # Extract feasible items
    w = remaining_weights[feasible_mask]
    v = remaining_values[feasible_mask]
    
    # Solve DP
    selected_local, total_val = _solve_dp(w, v, remaining_capacity, _SCALE)
    
    if len(selected_local) == 0:
        return -1
    
    # Map local index back to global index
    # Pick the first item in the selected set (or any). 
    # Picking the one with highest density might be safer for numerical stability?
    # Or just pick the first one found in backtrack (which is the last one added in DP order).
    # Let's pick the one with highest value-to-weight ratio among selected, just in case.
    # Actually, any item from the optimal set leads to the global optimum.
    # Let's just return the first one.
    idx_local = selected_local[0]
    idx_global = int(feasible_indices[idx_local])
    
    return idx_global
# EVOLVE-BLOCK-END
