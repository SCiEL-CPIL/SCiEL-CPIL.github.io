"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies + 1-swap refinement, returns best total value."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    mask = np.ones(n, dtype=bool)
    mask[excluded_idx] = False
    mask &= weights <= capacity
    if not np.any(mask):
        return 0.0

    idxs = np.where(mask)[0]
    w = weights[idxs]
    v = values[idxs]
    m = len(w)

    if m == 0:
        return 0.0

    best_total = 0.0
    best_selected = None  # track indices (into w/v) for swap refinement

    def _run_policy(order):
        total = 0.0
        cap = capacity
        selected = []
        for i in order:
            wi = w[i]
            if wi <= cap:
                total += v[i]
                cap -= wi
                selected.append(i)
                if cap <= 0:
                    break
        return total, selected

    # Policy 0: density
    order = np.argsort(-(v / w))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 1: value
    order = np.argsort(-v)
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 2: blend (density + 0.5 * value)
    order = np.argsort(-(v / w + 0.5 * v))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 3: small-weight first
    order = np.argsort(w)
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 4: snug-fit density (prefer items close to remaining capacity fraction)
    scores4 = (v / w) * (1 + 0.1 * (w / max(capacity, 1e-12)))
    order = np.argsort(-scores4)
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 5: value / sqrt(weight) — favors lighter high-value items
    order = np.argsort(-(v / np.sqrt(w)))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 6: value / weight^1.5 — strong light preference
    order = np.argsort(-(v / (w * np.sqrt(w))))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 7: light-first then density
    light_mask = w <= 0.3 * capacity
    if np.any(light_mask):
        light_order = np.where(light_mask)[0]
        light_order = light_order[np.argsort(-(v[light_order] / w[light_order]))]
        heavy_order = np.where(~light_mask)[0]
        heavy_order = heavy_order[np.argsort(-(v[heavy_order] / w[heavy_order]))]
        combined = np.concatenate([light_order, heavy_order])
        t, sel = _run_policy(combined)
        if t > best_total:
            best_total = t
            best_selected = sel

    # 1-swap refinement on the best solution found
    if best_selected is not None and len(best_selected) > 0 and m > len(best_selected):
        sel_set = set(best_selected)
        sel_arr = np.array(best_selected)
        unselected = np.setdiff1d(np.arange(m), sel_arr)
        if len(unselected) > 0:
            # Try replacing one selected item with one unselected item
            sel_w = w[sel_arr]
            sel_v = v[sel_arr]
            un_w = w[unselected]
            un_v = v[unselected]
            # For each selected item, find best unselected that fits in freed weight
            # Vectorized: for each selected i, unselected j can replace if un_w[j] <= sel_w[i]
            # and un_v[j] > sel_v[i]
            # We want max over pairs of (un_v[j] - sel_v[i]) subject to un_w[j] <= sel_w[i]
            # Do a simple loop over selected (usually small)
            for si in range(len(sel_arr)):
                freed_w = sel_w[si]
                freed_v = sel_v[si]
                # candidates: unselected with weight <= freed_w and value > freed_v
                cand_mask = (un_w <= freed_w + 1e-12) & (un_v > freed_v)
                if np.any(cand_mask):
                    cand_vals = un_v[cand_mask]
                    best_cand_local = int(np.argmax(cand_vals))
                    best_cand_val = cand_vals[best_cand_local]
                    new_val = best_total - freed_v + best_cand_val
                    if new_val > best_total:
                        best_total = new_val

    return best_total


def _build_value_function(capacity, weights, values, excluded_idx, num_buckets=200):
    """Build approximate DP value function via weight quantization.
    
    Returns: (value_func, bucket_width) where value_func[c] = max value achievable
    with capacity c*bucket_width using items not in excluded_idx.
    """
    n = len(weights)
    mask = np.ones(n, dtype=bool)
    if excluded_idx is not None:
        mask[excluded_idx] = False
    mask &= weights <= capacity
    if not np.any(mask):
        return None, None

    idxs = np.where(mask)[0]
    w = weights[idxs]
    v = values[idxs]
    m = len(w)
    
    if m == 0:
        return None, None

    # Quantize weights into buckets
    max_w = float(np.max(w))
    bucket_width = max_w / num_buckets
    if bucket_width < 1e-9:
        bucket_width = 1e-9
    
    # Scale weights to bucket indices
    w_buckets = np.clip(np.ceil(w / bucket_width).astype(int), 1, num_buckets)
    
    # DP: value_func[b] = max value achievable with capacity b*bucket_width
    value_func = np.zeros(num_buckets + 1, dtype=float)
    
    # Process items in order of decreasing density for better approximation
    density = v / w
    order = np.argsort(-density)
    
    for i in order:
        wi = int(w_buckets[i])
        vi = float(v[i])
        # Update DP backwards
        for b in range(num_buckets, wi - 1, -1):
            candidate = value_func[b - wi] + vi
            if candidate > value_func[b]:
                value_func[b] = candidate
    
    return value_func, bucket_width


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    elapsed = time.monotonic() - _START
    if elapsed > _TOTAL_BUDGET:
        return -1

    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    fn = len(fw)

    if fn == 1:
        return int(feasible_indices[0])

    # Fast fallback if time is tight
    if elapsed > _TOTAL_BUDGET * 0.7:
        density = fv / fw
        d_min, d_max = density.min(), density.max()
        v_min, v_max = fv.min(), fv.max()
        d_range = d_max - d_min + 1e-12
        v_range = v_max - v_min + 1e-12
        d_norm = (density - d_min) / d_range
        v_norm = (fv - v_min) / v_range
        scores = 0.6 * d_norm + 0.4 * v_norm
        return int(feasible_indices[int(np.argmax(scores))])

    # Medium fallback: simple lookahead with fewer items if time is moderate
    if elapsed > _TOTAL_BUDGET * 0.5:
        best_score = -1.0
        best_local_idx = 0
        for li in range(fn):
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                    feasible_indices[li])
            total_est = item_v + est
            if total_est > best_score:
                best_score = total_est
                best_local_idx = li
        return int(feasible_indices[best_local_idx])

    # Build value function for marginal gain scoring
    elapsed_check = time.monotonic() - _START
    if elapsed_check > _TOTAL_BUDGET * 0.8:
        # Too tight; use simple lookahead
        best_score = -1.0
        best_local_idx = 0
        for li in range(fn):
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                    feasible_indices[li])
            total_est = item_v + est
            if total_est > best_score:
                best_score = total_est
                best_local_idx = li
        return int(feasible_indices[best_local_idx])

    # Build approximate value function for current state
    value_func, bucket_width = _build_value_function(
        remaining_capacity, remaining_weights, remaining_values, None, num_buckets=200
    )
    
    if value_func is None:
        # Fallback to greedy lookahead
        best_score = -1.0
        best_local_idx = 0
        for li in range(fn):
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                    feasible_indices[li])
            total_est = item_v + est
            if total_est > best_score:
                best_score = total_est
                best_local_idx = li
        return int(feasible_indices[best_local_idx])

    # Score each candidate by marginal gain in value function
    # V(remaining_cap) - V(remaining_cap - item_w) approximates marginal value
    cap_bucket = min(int(np.ceil(remaining_capacity / bucket_width)), len(value_func) - 1)
    base_value = value_func[cap_bucket]
    
    scores_arr = np.zeros(fn)
    best_score = -1.0
    best_local_idx = 0
    
    for li in range(fn):
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        
        if remaining_cap < 0:
            scores_arr[li] = -np.inf
            continue
        
        rem_bucket = min(int(np.ceil(remaining_cap / bucket_width)), len(value_func) - 1)
        rem_value = value_func[rem_bucket]
        
        # Marginal gain: value from this item + expected value of remaining capacity
        marginal_gain = item_v + rem_value
        scores_arr[li] = marginal_gain
        
        if marginal_gain > best_score:
            best_score = marginal_gain
            best_local_idx = li

    # Also compute greedy lookahead scores for the top candidates to refine
    elapsed_check2 = time.monotonic() - _START
    if elapsed_check2 < _TOTAL_BUDGET * 0.85:
        # For top-5 candidates by value-function score, verify with greedy lookahead
        top_k = min(5, fn)
        top_candidates = np.argsort(-scores_arr)[:top_k]
        
        for li in top_candidates:
            if scores_arr[li] == -np.inf:
                continue
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                    feasible_indices[li])
            total_est = item_v + est
            if total_est > best_score:
                best_score = total_est
                best_local_idx = li

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
