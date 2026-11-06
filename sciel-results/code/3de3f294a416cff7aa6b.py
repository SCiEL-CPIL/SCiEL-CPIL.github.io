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
        zero_indices = np.where(zero_w_mask)[0]
        return int(feasible_indices[zero_indices[0]])

    # Compute densities
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = 0.0

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

    # Strategy 4: DP based selection
    if time.time() < deadline:
        # Determine scale factor
        target_cap_int = 3000
        scale = target_cap_int / C

        w_int = np.round(w_f * scale).astype(int)

        C_int = int(np.round(C * scale))

        # Safety check: if C_int is too large, skip DP
        if C_int <= 10000 and n_f <= 500:
            # dp[c] = max value achievable with capacity c
            dp = np.zeros(C_int + 1, dtype=float)
            # parent[c] = index of the item used to reach dp[c] from dp[c - w_i]
            parent = np.full(C_int + 1, -1, dtype=int)

            for i in range(n_f):
                wi = w_int[i]
                vi = v_f[i]
                if wi == 0:
                    if vi > 0:
                        dp[:] += vi
                else:
                    for c in range(C_int, wi - 1, -1):
                        new_val = dp[c - wi] + vi
                        if new_val > dp[c]:
                            dp[c] = new_val
                            parent[c] = i

            # If we have items that were rounded to 0 weight, we should pick one of them first
            zero_int_mask = (w_int == 0) & (v_f > 0)
            if np.any(zero_int_mask):
                zero_int_indices = np.where(zero_int_mask)[0]
                if dp[C_int] > best_score:
                    best_score = dp[C_int]
                    best_i_local = zero_int_indices[0]
            else:
                # Reconstruct solution
                c = C_int
                chosen_item = -1
                while c > 0 and parent[c] != -1:
                    chosen_item = parent[c]
                    c -= w_int[chosen_item]

                if chosen_item != -1:
                    if dp[C_int] > best_score:
                        best_score = dp[C_int]
                        best_i_local = chosen_item

    if best_i_local == -1:
        return -1

    return int(feasible_indices[best_i_local])
