import numpy as np
import time

_START = None
_HARD_LIMIT = 145.0  # Safety margin before 150s kill

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _dp_solve(cap, weights, values):
    """
    Solves the 0/1 knapsack problem using value-scaling DP.
    Returns the set of selected indices that form the optimal (or near-optimal) solution.
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool)
    
    # Filter out infeasible items immediately to reduce DP size
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return np.zeros(n, dtype=bool)
    
    # We work on the full array but ignore infeasible items in DP updates
    # To keep DP table size manageable, we scale values.
    # Target max scaled value ~ 2000-5000 to keep runtime low (n * V * 100 items = 100 * 5000 = 500k ops)
    # Let's aim for a scale such that sum(scaled_values) <= 5000.
    
    total_val = np.sum(values)
    if total_val <= 0:
        return np.zeros(n, dtype=bool)
    
    # Heuristic for scaling factor
    # If values are large, scale down.
    # We want the number of DP states to be small.
    # Let's try to map total value to ~ 2000 units.
    # scale = 2000 / total_val
    # But we need integer scaling.
    # Let's check magnitude of values.
    max_val = np.max(values)
    min_val = np.min(values)
    
    # Adaptive scaling
    # If max_val is small, we can use finer granularity.
    # If max_val is large, we must scale more.
    # Let's target a DP table size of roughly 1000-2000 to be safe within time limits for 8 instances.
    # 100 items * 2000 states = 200,000 operations per DP run.
    # Each operation is O(1) with numpy vectorization? No, standard DP is O(n*V).
    # With Python loops, 200k ops is fast (~0.01s).
    # We can afford V up to ~5000.
    
    target_v_states = 4000
    scale = target_v_states / total_val
    
    # Ensure scale is not too small (causing all values to become 0)
    if scale < 1e-6:
        scale = 1e-6
        
    scaled_values = np.floor(values * scale).astype(np.int64)
    
    # If some values are still 0 but original > 0, we might lose distinction.
    # To handle this, if min(scaled_values[scaled_values>0]) == 0, we might need to adjust.
    # But for simplicity, we proceed. Items with 0 scaled value are treated as "free" in value terms 
    # but still consume weight. This is a known approximation error.
    
    max_scaled_val = np.sum(scaled_values)
    
    # DP Table: dp[v] = min weight to achieve scaled value v
    # Initialize with infinity
    INF = np.iinfo(np.int64).max // 2
    dp = np.full(max_scaled_val + 1, INF, dtype=np.int64)
    dp[0] = 0
    
    # Track parent pointers to reconstruct solution
    # parent[v] = (i, prev_v) indicating item i was added to reach v from prev_v
    # We only need the best path.
    # Storing full parent array of size (n * V) is memory heavy.
    # Alternative: Store only the last item index for each state, and backtrack.
    # But standard 0/1 knapsack DP doesn't support easy backtracking without storing the choice per state per item.
    
    # Since n=100 and V=4000, a table of shape (n, V) is 400,000 entries.
    # Each entry can be a bool (1 if item i is used to reach state v).
    # This is 400k bytes. Very manageable.
    
    # choice[i][v] = True if item i is part of the optimal solution for value v (considering items 0..i)
    # Actually, standard DP:
    # dp[i][v] = max value using items 0..i with weight <= cap? No, we are minimizing weight for value.
    # Let's use:
    # dp[v] = min weight to achieve scaled value v using items processed so far.
    # To backtrack, we need to know which items were chosen.
    # We can store a 2D array `take` of shape (n, max_scaled_val + 1).
    # take[i][v] is True if item i is included in the optimal solution for scaled value v.
    
    take = np.zeros((n, max_scaled_val + 1), dtype=bool)
    
    # We iterate items 0 to n-1
    # We update dp in reverse order of values to ensure 0/1 property.
    
    for i in range(n):
        if not feasible_mask[i]:
            continue
            
        wi = weights[i]
        vi_scaled = scaled_values[i]
        
        if vi_scaled == 0:
            # If scaled value is 0, we can only add it if it doesn't break weight constraint 
            # and if it doesn't reduce the "value" (which it doesn't). 
            # But for DP on value, adding a 0-value item doesn't change v.
            # It effectively competes with other 0-value items for weight space.
            # This case is tricky. Usually, we'd want to include 0-value items only if we have spare weight 
            # and it doesn't block higher value items. But since we are minimizing weight for a fixed value,
            # adding a 0-value item increases weight for the same value, so it will never be chosen 
            # in the "min weight for value" DP unless it's the only way to achieve that value (which is v=0).
            # So we can skip 0-value items in the DP update for v > 0.
            # For v=0, we don't care about weight as long as it's 0? No, dp[0]=0.
            # So 0-value items are effectively ignored by this DP formulation.
            # This is a limitation. If there are many 0-scaled items, they are ignored.
            # This might hurt if "low value" items are important fillers.
            # However, since we scale based on total value, "low value" items might have small but non-zero scaled values.
            # Only if value * scale < 1 do they become 0.
            continue
            
        # Update dp in reverse
        # for v from max_scaled_val down to vi_scaled:
        #    if dp[v - vi_scaled] + wi < dp[v]:
        #        dp[v] = dp[v - vi_scaled] + wi
        #        take[i][v] = True
        
        # Vectorized update is hard because of the dependency on dp[v-vi].
        # But n=100, V=4000. 400k iterations in Python is ~0.02s.
        # 100 * 0.02s = 2s per instance. 8 instances = 16s. Acceptable.
        
        for v in range(max_scaled_val, vi_scaled - 1, -1):
            prev_w = dp[v - vi_scaled]
            if prev_w != INF:
                new_w = prev_w + wi
                if new_w < dp[v]:
                    dp[v] = new_w
                    take[i][v] = True
                # Note: if new_w == dp[v], we might have a tie. 
                # We prefer the solution that uses fewer items or lighter items? 
                # For backtracking consistency, we just keep the first found (which is the one from higher index i? 
                # No, we iterate i increasing. So later items overwrite earlier if equal? 
                # Actually, if new_w < dp[v], we update. If equal, we don't.
                # This means we prefer the solution that was found earlier (with smaller i).
    
    # Find the best achievable scaled value within capacity
    # We want max v such that dp[v] <= cap
    best_v = 0
    for v in range(max_scaled_val, -1, -1):
        if dp[v] <= cap:
            best_v = v
            break
            
    # Backtrack to find selected items
    selected = np.zeros(n, dtype=bool)
    current_v = best_v
    for i in range(n - 1, -1, -1):
        if current_v > 0 and take[i][current_v]:
            selected[i] = True
            current_v -= scaled_values[i]
        # If current_v == 0, no more value to recover.
    
    return selected

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    if n == 0 or cap <= 0:
        return -1
        
    if not np.any(weights <= cap):
        return -1

    # Check time
    if _now() - _START > _HARD_LIMIT:
        # Fallback: fastest greedy
        feasible = weights <= cap
        if not np.any(feasible):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        fd = np.where(feasible, density, -np.inf)
        return int(np.argmax(fd))

    # Solve with DP
    best_sel = _dp_solve(cap, weights, values)
    
    selected_indices = np.where(best_sel)[0]
    
    if len(selected_indices) == 0:
        # Fallback if DP failed or selected nothing (e.g. all scaled to 0)
        feasible = weights <= cap
        if not np.any(feasible):
            return -1
        # Pick max value feasible
        feasible_vals = values[feasible]
        feasible_idx = np.where(feasible)[0]
        return int(feasible_idx[np.argmax(feasible_vals)])
        
    # Strategy: Pick the item with the HIGHEST VALUE from the optimal set.
    # Why? 
    # 1. It is part of the optimal solution.
    # 2. Removing it leaves a subproblem that is likely still well-solved by the next DP call 
    #    (since the DP will re-optimize the remaining capacity).
    # 3. High value items are "anchor" items. Securing them early reduces the risk of 
    #    "wasting" capacity on lower value items in subsequent steps if the heuristic were less robust.
    # 4. Alternatively, pick the item with the LOWEST "regret". 
    #    But since we re-solve the whole DP each time, the "regret" is handled by the global re-optimization.
    #    So the choice of which item to pick *first* from the set matters less for the final sum, 
    #    BUT it matters for the *path*. 
    #    If we pick a high value item, we reduce cap significantly. 
    #    If we pick a low value item, we reduce cap slightly.
    #    Does it matter? 
    #    The DP is re-run on the *remaining* items and *remaining* capacity.
    #    So the set of items available changes.
    #    If we pick item X, X is removed from the pool.
    #    The new optimal solution for the remaining pool might be different from (Optimal_Solution - {X}).
    #    We want to pick X such that the new optimal solution is as close to the original optimal value as possible.
    #    This is equivalent to finding the item in the optimal set whose removal causes the least drop in optimal value.
    #    This is a "core" concept.
    #    However, computing this requires running DP n times (once for each removal).
    #    n=100, DP is ~0.02s. 100 * 0.02s = 2s per step. 
    #    Steps ~ 10-20? 20 * 2s = 40s. Total 8 instances * 40s = 320s. TOO SLOW.
    #    
    #    So we must use a proxy.
    #    Proxy 1: Highest Value. (Current choice)
    #    Proxy 2: Lowest Weight. (Preserves capacity for others)
    #    Proxy 3: Highest Density.
    #    
    #    Let's stick with Highest Value. It's a strong indicator of importance.
    
    vals_in_sel = values[selected_indices]
    best_local_idx = np.argmax(vals_in_sel)
    return int(selected_indices[best_local_idx])
