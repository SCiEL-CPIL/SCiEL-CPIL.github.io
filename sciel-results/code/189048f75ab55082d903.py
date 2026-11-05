import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 140.0

def _solve_knapsack_dp(w, v, C):
    """
    Solves 0/1 knapsack exactly using DP.
    Returns a list of indices of the items in the optimal solution.
    """
    n = len(w)
    if n == 0 or C <= 0:
        return []

    # Scale weights to integers to use as array indices
    # Capacity is ~12.5, scale by 10000 -> 125000.
    # Higher precision to avoid rounding errors affecting optimality.
    scale = 10000
    cap_int = int(round(C * scale))
    
    if cap_int < 0:
        return []

    # dp[c] = max value with capacity c
    dp = np.zeros(cap_int + 1, dtype=np.float64)
    
    # take[i, c] = True if item i is taken in the optimal solution for capacity c
    # Using a 2D boolean array for backtracking
    take = np.zeros((n, cap_int + 1), dtype=bool)
    
    # Fill DP table
    for i in range(n):
        wi = int(round(w[i] * scale))
        vi = v[i]
        
        if wi > cap_int:
            # Item too heavy, cannot be taken
            # take[i, :] remains False
            continue
            
        if wi == 0:
            # Item has 0 weight, always take if value > 0
            if vi > 0:
                dp += vi
                take[i, :] = True
            continue

        # Vectorized update for 1D DP
        # We need to be careful with the direction or use a copy
        # dp[wi:] = max(dp[wi:], dp[:-wi] + vi)
        # But we also need to record 'take'
        
        # Calculate values if we take item i
        # For c from wi to cap_int:
        # candidate = dp[c - wi] + vi
        # We compare with dp[c]
        
        # Note: dp[:-wi] corresponds to dp[0] to dp[cap_int-wi]
        # dp[wi:] corresponds to dp[wi] to dp[cap_int]
        
        candidates = dp[:cap_int - wi + 1] + vi
        current = dp[wi:]
        
        # Determine where taking is better
        # If candidates > current, we take
        # Note: In case of tie, we prefer NOT taking to keep capacity? 
        # Or does it matter? 
        # For backtracking consistency, if we prefer not taking on tie, 
        # take[i, c] should be False.
        take_mask = candidates > current
        
        # Update take array
        take[i, wi:] = take_mask
        
        # Update DP values
        # np.where(take_mask, candidates, current)
        # Or simply:
        # dp[wi:] = np.maximum(current, candidates)
        # But we must use the OLD dp values for candidates.
        # The slice dp[:cap_int - wi + 1] is a view, but we are reading from it.
        # If we update dp in place, we might overwrite values needed for later c?
        # In 1D DP, we usually iterate backwards.
        # Here, we are doing a vectorized forward pass?
        # Wait.
        # dp_new[c] = max(dp_old[c], dp_old[c-wi] + vi)
        # If we do dp[wi:] = np.maximum(dp[wi:], dp[:-wi] + vi)
        # The RHS dp[:-wi] is evaluated BEFORE the assignment to LHS dp[wi:]?
        # In NumPy, `dp[:-wi]` creates a view or copy?
        # If it's a view, and we assign to `dp[wi:]`, do they overlap?
        # `dp[:-wi]` is indices 0 to cap_int-wi.
        # `dp[wi:]` is indices wi to cap_int.
        # They overlap if wi < cap_int - wi + 1?
        # Example: cap=10, wi=2.
        # dp[:-2] is 0..8.
        # dp[2:] is 2..10.
        # Overlap is 2..8.
        # If we update dp[2] first, does it affect dp[4] calculation?
        # dp[4] uses dp[4-2]=dp[2].
        # If dp[2] is updated, dp[4] uses the NEW dp[2].
        # This is the 0/1 knapsack trap. We must use OLD values.
        
        # So we MUST use a copy of the previous state or iterate backwards.
        # Vectorized backwards is hard.
        # Let's use a temporary array for the "take" candidates or copy the row.
        
        # Actually, `dp[:cap_int - wi + 1]` is the source.
        # If we do `temp = dp[:cap_int - wi + 1].copy()`, then `candidates = temp + vi`.
        # Then update.
        
        # Let's just copy the relevant part of dp to ensure we use old values.
        # Since cap is small (12500), copying is cheap.
        
        # Optimization: Only copy the part we need?
        # dp_old = dp.copy()
        # candidates = dp_old[:cap_int - wi + 1] + vi
        # current = dp[wi:]
        # take_mask = candidates > current
        # take[i, wi:] = take_mask
        # dp[wi:] = np.maximum(current, candidates)
        
        # This is O(C) copy. 100 items * 12500 = 1.25M floats.
        # 1.25M * 8 bytes = 10 MB.
        # 100 iterations -> 1 GB memory traffic.
        # Should be fast enough.

        dp_old = dp.copy()
        candidates = dp_old[:cap_int - wi + 1] + vi
        current = dp[wi:]
        
        take_mask = candidates > current
        take[i, wi:] = take_mask
        
        dp[wi:] = np.maximum(current, candidates)

    # Backtrack to find selected items
    selected_indices = []
    c = cap_int
    for i in range(n - 1, -1, -1):
        if take[i, c]:
            selected_indices.append(i)
            c -= int(round(w[i] * scale))
            
    selected_indices.reverse()
    return selected_indices

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        if len(remaining_weights) == 0:
            return -1
        # Fallback: Simple greedy
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
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
    
    # Run DP to get optimal set
    try:
        optimal_indices = _solve_knapsack_dp(w, v, C)
    except Exception:
        # Fallback on error
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0:
            return -1
        densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    if not optimal_indices:
        return -1
        
    # From the optimal set, pick the item that is most "critical" to the solution.
    # An item is critical if removing it causes the largest loss in total value.
    # This is approximated by: value / weight (density) but we also consider
    # whether the item is "essential" - i.e., there are few alternatives.
    # Strategy: pick the item with highest density among the optimal set,
    # but break ties by preferring items that are harder to replace (lower weight
    # relative to alternatives in the optimal set).
    
    sel_w = w[optimal_indices]
    sel_v = v[optimal_indices]
    n_sel = len(optimal_indices)
    
    if n_sel == 1:
        return int(optimal_indices[0])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sel_densities = sel_v / sel_w
    sel_densities = np.nan_to_num(sel_densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Score each item: primary key is density, secondary key is "criticality"
    # Criticality: items with weight close to the minimum weight in the set
    # are more likely to be irreplaceable (harder to find a lighter substitute)
    min_w = np.min(sel_w)
    if min_w > 0:
        # Items with lower weight relative to min are more "flexible"
        # Items with higher weight are more "committed"
        # We want to pick items that are both high-value and committed
        weight_factor = sel_w / min_w  # 1.0 for lightest, >1 for heavier
        # Combine: score = density * weight_factor
        # This favors items that are dense AND relatively heavy (committed)
        scores = sel_densities * weight_factor
    else:
        scores = sel_densities.copy()
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    best_pick_rel = np.argmax(scores)
    best_idx = optimal_indices[best_pick_rel]
    
    return int(best_idx)
