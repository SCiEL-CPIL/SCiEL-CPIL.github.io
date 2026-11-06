# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 130.0  # safety margin before 150s

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    if remaining_capacity <= 0:
        return -1
    
    # Fast fallback: highest density
    if _time_left() < 5.0:
        densities = remaining_values / np.maximum(remaining_weights, 1e-12)
        return int(np.argmax(densities))
    
    # Scale capacity to integer for DP
    scale = 10
    cap_int = int(round(remaining_capacity * scale))
    weights_int = np.array([int(round(w * scale)) for w in remaining_weights])
    values = np.array(remaining_values, dtype=float)
    
    # Filter feasible items
    feasible = weights_int <= cap_int
    if not np.any(feasible):
        return -1
    
    # DP: dp[c] = max value achievable with capacity c
    # dp[c][i] -> use 2D or 1D with reconstruction
    # For reconstruction, use 2D: dp[i][c]
    n_f = int(np.sum(feasible))
    if n_f == 0:
        return -1
    
    # Map feasible indices back to original
    feasible_indices = np.where(feasible)[0]
    fw = weights_int[feasible]
    fv = values[feasible]
    
    # DP table: dp[i][c] = max value using first i items with capacity c
    # dp[0][c] = 0
    # dp[i][c] = max(dp[i-1][c], dp[i-1][c-fw[i-1]] + fv[i-1]) if fw[i-1] <= c
    # For reconstruction, store choice
    
    # Use 1D DP for speed, but need reconstruction -> use 2D
    # n_f * (cap_int+1) booleans for choice
    # cap_int <= 125, n_f <= 100 -> 12500 entries, fine
    
    dp = np.zeros((n_f + 1, cap_int + 1), dtype=float)
    take = np.zeros((n_f + 1, cap_int + 1), dtype=bool)
    
    for i in range(1, n_f + 1):
        w = fw[i - 1]
        v = fv[i - 1]
        for c in range(cap_int + 1):
            if w <= c and dp[i-1][c - w] + v > dp[i-1][c]:
                dp[i][c] = dp[i-1][c - w] + v
                take[i][c] = True
            else:
                dp[i][c] = dp[i-1][c]
    
    # Reconstruct which items are taken
    taken = np.zeros(n_f, dtype=bool)
    c = cap_int
    for i in range(n_f, 0, -1):
        if take[i][c]:
            taken[i - 1] = True
            c -= fw[i - 1]
    
    if not np.any(taken):
        # Fallback to density
        densities = remaining_values / np.maximum(remaining_weights, 1e-12)
        return int(np.argmax(densities))
    
    # Among taken items, pick the one with highest value (or density)
    taken_fv = fv[taken]
    best_local = int(np.argmax(taken_fv))
    # Map back to feasible_indices
    taken_indices = feasible_indices[taken]
    best_global = int(taken_indices[best_local])
    
    return best_global
# EVOLVE-BLOCK-END
