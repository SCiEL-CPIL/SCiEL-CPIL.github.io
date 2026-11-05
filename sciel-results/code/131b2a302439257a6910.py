"""Combined program: High-precision DP with robust scaling and tie-breaking."""
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s limit

def _solve_dp(w, v, C, scale):
    """
    Solves 0/1 knapsack exactly using DP on scaled integer weights.
    Returns (list of indices in optimal solution, total value).
    """
    n = len(w)
    if n == 0:
        return [], 0.0
        
    cap_int = int(round(C * scale))
    if cap_int < 0:
        return [], 0.0

    # We need to track choices for backtracking.
    # For N=100, C=12500 (scale=1000), a 2D table is 100*12501 ~ 1.25M floats.
    # This is ~10MB. Manageable.
    
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    
    # Pre-compute integer weights
    w_int = np.round(w * scale).astype(np.int64)
    w_int = np.maximum(w_int, 0) # Safety for negative weights (shouldn't happen)

    for i in range(1, n + 1):
        wi = w_int[i-1]
        vi = v[i-1]
        
        if wi <= cap_int:
            # Copy previous row
            dp[i, :] = dp[i-1, :]
            
            if wi > 0:
                # Update where c >= wi
                # dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
                take_vals = dp[i-1, 0:cap_int-wi+1] + vi
                prev_vals = dp[i-1, wi:cap_int+1]
                
                dp[i, wi:cap_int+1] = np.maximum(prev_vals, take_vals)
            else:
                # wi == 0
                # If vi > 0, we always take it, so add vi to all capacities
                if vi > 0:
                    dp[i, :] += vi
        else:
            # Item too heavy, copy previous
            dp[i, :] = dp[i-1, :]

    # Backtrack to find selected items
    selected = []
    c = cap_int
    for i in range(n, 0, -1):
        # Check if item i-1 was taken
        # If dp[i, c] > dp[i-1, c], it was taken
        # We use a small epsilon to handle floating point precision issues
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected.append(i - 1)
            wi = w_int[i-1]
            c -= wi
            if c < 0:
                break
        # else: not taken

    selected.reverse()
    return selected, dp[n, cap_int]

def _fallback_greedy(w, v, C):
    """
    Fallback heuristic: pick item with highest density (v/w) among feasible items.
    """
    if len(w) == 0:
        return -1
    w = np.array(w, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    feasible = np.where(w <= C)[0]
    if len(feasible) == 0:
        return -1
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v[feasible] / w[feasible]
    densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
    best_local = np.argmax(densities)
    return int(feasible[best_local])

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        return _fallback_greedy(remaining_weights, remaining_values, remaining_capacity)

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
    
    # Extract feasible items for DP
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    n_feas = len(w_feas)
    
    if n_feas == 0:
        return -1

    # Determine scale for integer weights
    # Scale 1000 was found to improve accuracy over 100.
    # C=12.5 -> cap_int=12500.
    # DP size: 100 * 12501 ~ 1.25M floats.
    # Time per call: ~10-20ms in NumPy.
    # 8 instances * 100 steps = 800 calls.
    # Total time: ~8-16 seconds. Well within 150s limit.
    
    scale = 1000
    cap_int = int(round(C * scale))
    
    if cap_int < 1:
        # If capacity is very small, just pick the lightest feasible item
        lightest_idx = np.argmin(w_feas)
        return int(feasible_indices[lightest_idx])

    if cap_int > 50000: 
        # Fallback to heuristic if capacity is huge (unlikely given C=12.5)
        return _fallback_greedy(w_feas, v_feas, C)

    try:
        selected_indices, total_value = _solve_dp(w_feas, v_feas, C, scale)
    except Exception:
        # Fallback on error
        return _fallback_greedy(w_feas, v_feas, C)

    if not selected_indices:
        # If no items selected in optimal solution (e.g., all values <= 0), stop?
        # The task implies picking items to maximize value. If optimal is empty, stop.
        return -1
        
    # Pick the first item in the optimal set?
    # Or the one with highest density?
    # It doesn't matter for the total value, as long as we pick from the optimal set.
    # However, picking a "small" item might leave more capacity for later, but since we re-solve DP,
    # the future decisions will adapt.
    # Let's just pick the first one.
    # Actually, let's pick the one with the highest value? Or density?
    # It shouldn't matter. Let's pick the one with index 0 in the sorted list of selected items?
    # The selected_indices list is in increasing index order (from backtracking).
    # Let's just pick the first one.
    
    return int(feasible_indices[selected_indices[0]])

# EVOLVE-BLOCK-END
