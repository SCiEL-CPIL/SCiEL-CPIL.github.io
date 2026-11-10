import numpy as np
import time

_start_time = None
_GLOBAL_DEADLINE = 140.0

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_time_up():
    return _get_elapsed() > _GLOBAL_DEADLINE

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if _is_time_up():
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)

    # Handle zero weight positive value items first
    zero_weight_indices = np.where((weights == 0) & (values > 0))[0]
    if len(zero_weight_indices) > 0:
        return int(zero_weight_indices[0])

    # Filter feasible items (weight > 0 and weight <= capacity)
    feasible_mask = (weights <= remaining_capacity) & (weights > 0)
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)

    if f_n == 0:
        return -1

    C = remaining_capacity

    # Try DP if small enough, otherwise use fractional surrogate
    min_w = np.min(f_weights)
    if min_w <= 0:
        return -1

    # Calculate required scale to avoid rounding min_w to 0
    scale = int(1.0 / min_w) + 1
    C_scaled = int(C * scale)

    MAX_C_SCALED = 150_000
    MAX_OPS = 4_000_000

    if C_scaled > MAX_C_SCALED:
        new_scale = int(MAX_C_SCALED / C)
        if new_scale > 0 and min_w * new_scale >= 1:
            scale = new_scale
            C_scaled = int(C * scale)
        else:
            return _solve_fractional_surrogate(f_values, f_weights, C, feasible_indices, f_n)

    w_scaled = (f_weights * scale).astype(np.int64)

    dp_size = f_n * (C_scaled + 1)

    if dp_size < MAX_OPS and C_scaled < MAX_C_SCALED:
        return _solve_dp(f_values, w_scaled, C_scaled, feasible_indices, f_n)
    else:
        return _solve_fractional_surrogate(f_values, f_weights, C, feasible_indices, f_n)

def _solve_dp(values, weights, capacity, original_indices, n):
    dp = np.zeros(capacity + 1, dtype=np.float64)
    takes = np.zeros((n, capacity + 1), dtype=bool)

    for i in range(n):
        w = weights[i]
        v = values[i]
        if w == 0:
            continue

        candidate = dp[:-w] + v
        mask = candidate > dp[w:]

        if np.any(mask):
            dp[w:][mask] = candidate[mask]
            takes[i][w:][mask] = True

    j = capacity
    selected_f_indices = []

    for i in range(n - 1, -1, -1):
        if takes[i][j]:
            selected_f_indices.append(i)
            j -= weights[i]
            if j < 0:
                break

    if not selected_f_indices:
        best_val = -1
        best_idx = -1
        for idx in range(n):
            if weights[idx] <= capacity:
                if values[idx] > best_val:
                    best_val = values[idx]
                    best_idx = idx
        if best_idx == -1:
            return -1
        return int(original_indices[best_idx])

    f_idx_to_pick = selected_f_indices[0]
    orig_idx = original_indices[f_idx_to_pick]

    return int(orig_idx)

def _solve_fractional_surrogate(values, weights, capacity, original_indices, n):
    densities = values / weights
    order = np.argsort(-densities)

    sorted_values = values[order]
    sorted_weights = weights[order]
    sorted_orig_indices = original_indices[order]

    remaining_cap = capacity
    fully_taken_mask = np.zeros(n, dtype=bool)
    fractional_idx = -1

    for i in range(n):
        w = sorted_weights[i]
        if remaining_cap <= 0:
            break

        if w <= remaining_cap:
            fully_taken_mask[i] = True
            remaining_cap -= w
        else:
            fractional_idx = i
            remaining_cap = 0
            break

    candidate_indices_in_sorted = np.where(fully_taken_mask)[0]
    if fractional_idx != -1:
        candidate_indices_in_sorted = np.append(candidate_indices_in_sorted, fractional_idx)

    if len(candidate_indices_in_sorted) == 0:
        return -1

    candidate_densities = densities[candidate_indices_in_sorted]
    best_local_idx = np.argmax(candidate_densities)
    best_sorted_idx = candidate_indices_in_sorted[best_local_idx]

    return int(sorted_orig_indices[best_sorted_idx])

def _solve_bb(values, weights, capacity, original_indices, n):
    densities = values / weights
    order = np.argsort(-densities)

    sorted_values = values[order]
    sorted_weights = weights[order]
    sorted_orig_indices = original_indices[order]

    prefix_values = np.cumsum(sorted_values)
    prefix_weights = np.cumsum(sorted_weights)

    best_val = 0.0
    best_set = set()

    start_time = time.time()
    time_limit = 0.5

    def dfs(i, curr_val, curr_weight, curr_set):
        nonlocal best_val, best_set
        if time.time() - start_time > time_limit:
            return

        if i == n:
            if curr_val > best_val:
                best_val = curr_val
                best_set = set(curr_set)
            return

        rem_cap = capacity - curr_weight
        if rem_cap <= 0:
            if curr_val > best_val:
                best_val = curr_val
                best_set = set(curr_set)
            return

        sum_rem_values = prefix_values[-1] - (prefix_values[i-1] if i > 0 else 0)
        sum_rem_weights = prefix_weights[-1] - (prefix_weights[i-1] if i > 0 else 0)

        if sum_rem_weights <= rem_cap:
            ub = curr_val + sum_rem_values
        else:
            rem_w = sorted_weights[i:]
            rem_v = sorted_values[i:]
            cum_w = np.cumsum(rem_w)
            k = np.searchsorted(cum_w, rem_cap, side='right') - 1

            if k < 0:
                ub = curr_val + rem_v[0] * (rem_cap / rem_w[0])
            else:
                val_full = np.sum(rem_v[:k+1])
                if k+1 < len(rem_w):
                    frac_val = rem_v[k+1] * (rem_cap - cum_w[k]) / rem_w[k+1]
                else:
                    frac_val = 0.0
                ub = curr_val + val_full + frac_val

        if ub <= best_val:
            return

        v_i = sorted_values[i]
        w_i = sorted_weights[i]
        idx_i = sorted_orig_indices[i]

        if curr_weight + w_i <= capacity:
            curr_set.add(idx_i)
            dfs(i + 1, curr_val + v_i, curr_weight + w_i, curr_set)
            curr_set.remove(idx_i)

        dfs(i + 1, curr_val, curr_weight, curr_set)

    curr_set = set()
    dfs(0, 0.0, 0.0, curr_set)

    if not best_set:
        feasible = sorted_weights <= capacity
        if not np.any(feasible):
            return -1
        best_idx_in_sorted = np.where(feasible)[0][0]
        return int(sorted_orig_indices[best_idx_in_sorted])

    best_item_idx = None
    for idx in best_set:
        best_item_idx = idx
        break

    return int(best_item_idx)
