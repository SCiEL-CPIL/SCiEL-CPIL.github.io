import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    start_time = time.time()
    deadline = start_time + 100.0  # 100s margin out of 150s total
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)
    
    if C <= 0:
        return -1

    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)
    
    if n_f == 0:
        return -1

    # Check for zero-weight items with positive value
    zero_w_mask = (w_f == 0) & (v_f > 0)
    if np.any(zero_w_mask):
        # Pick the first one (or any one)
        zero_indices = np.where(zero_w_mask)[0]
        return int(feasible_indices[zero_indices[0]])

    # Compute densities
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = 0.0 # Zero weight items with 0 value are useless
    
    # Sort by density descending
    sort_idx = np.argsort(-d_f)
    w_s = w_f[sort_idx]
    v_s = v_f[sort_idx]
    d_s = d_f[sort_idx]
    
    # Prefix sums for LP bound
    pw = np.cumsum(w_s)
    pv = np.cumsum(v_s)
    
    def get_lp_value(cap):
        if cap <= 1e-9:
            return 0.0
        k = np.searchsorted(pw, cap, side='right') - 1
        if k < 0:
            if len(w_s) == 0:
                return 0.0
            return cap * d_s[0]
        val = pv[k]
        remaining_cap = cap - pw[k]
        if k < len(w_s) - 1:
            next_w = w_s[k + 1]
            next_d = d_s[k + 1]
            val += remaining_cap * next_d
        return val

    # Strategy 1: Greedy by density
    # Strategy 2: Greedy by value
    # Strategy 3: Greedy by weight (lightest first)
    
    best_score = -1.0
    best_i_local = -1
    
    # Helper to simulate a greedy packing and compute total value
    def simulate_greedy(capacity, weights, values, order):
        total_value = 0.0
        current_cap = capacity
        for idx in order:
            if weights[idx] > current_cap:
                continue
            total_value += values[idx]
            current_cap -= weights[idx]
            if current_cap <= 1e-9:
                break
        return total_value

    # Strategy 1: Density order
    density_order = sort_idx
    if time.time() < deadline:
        score1 = simulate_greedy(C, w_f, v_f, density_order)
        if score1 > best_score:
            best_score = score1
            best_i_local = density_order[0]
    
    # Strategy 2: Value order
    if time.time() < deadline:
        value_order = np.argsort(-v_f)
        score2 = simulate_greedy(C, w_f, v_f, value_order)
        if score2 > best_score:
            best_score = score2
            best_i_local = value_order[0]
    
    # Strategy 3: Weight order (lightest first)
    if time.time() < deadline:
        weight_order = np.argsort(w_f)
        score3 = simulate_greedy(C, w_f, v_f, weight_order)
        if score3 > best_score:
            best_score = score3
            best_i_local = weight_order[0]
            
    # Strategy 4: Value-scaling DP based selection
    if time.time() < deadline:
        # Value-scaling: scale values to integers, minimize weight for each value level
        v_max = np.max(v_f) if n_f > 0 else 0
        if v_max > 0:
            target_v_int = 3000
            v_scale = target_v_int / v_max
            v_int = np.round(v_f * v_scale).astype(int)
            
            # Ensure no zero-value items with positive value get rounded to 0
            # If v_int[i] == 0 and v_f[i] > 0, set to 1
            zero_v_mask = (v_int == 0) & (v_f > 0)
            v_int[zero_v_mask] = 1
            
            V_int = int(np.round(v_max * v_scale))
            
            # Safety check
            if V_int <= 10000 and n_f <= 500:
                # dp[val] = min weight to achieve at least 'val' value
                INF = float('inf')
                dp = np.full(V_int + 1, INF, dtype=float)
                dp[0] = 0.0
                # parent[val] = index of the item that achieved dp[val]
                parent = np.full(V_int + 1, -1, dtype=int)
                
                for i in range(n_f):
                    wi = w_f[i]
                    vi = v_int[i]
                    if vi == 0:
                        continue
                    # Update in reverse to avoid using same item twice
                    for val in range(V_int, vi - 1, -1):
                        new_w = dp[val - vi] + wi
                        if new_w < dp[val] and dp[val - vi] < INF:
                            dp[val] = new_w
                            parent[val] = i
                
                # Find the maximum value achievable within capacity C
                # dp[val] <= C means we can achieve at least 'val' value
                feasible_vals = np.where(dp <= C + 1e-9)[0]
                if len(feasible_vals) > 0:
                    best_val = feasible_vals[-1]  # highest achievable value level
                    # Reconstruct: find the last item added
                    val = best_val
                    chosen_item = -1
                    while val > 0 and parent[val] != -1:
                        chosen_item = parent[val]
                        val -= v_int[chosen_item]
                    
                    if chosen_item != -1:
                        # The DP says we can achieve value level best_val
                        # The actual value is approximately best_val / v_scale
                        est_value = best_val / v_scale
                        if est_value > best_score:
                            best_score = est_value
                            best_i_local = chosen_item

    if best_i_local == -1:
        return -1
    
    return int(feasible_indices[best_i_local])
