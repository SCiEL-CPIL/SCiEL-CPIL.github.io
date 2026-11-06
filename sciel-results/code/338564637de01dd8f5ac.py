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

    # Full lookahead: for each feasible item, estimate total via greedy completion
    best_score = -1.0
    best_local_idx = 0
    scores_arr = np.zeros(fn)

    for li in range(fn):
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est
        scores_arr[li] = total_est
        if total_est > best_score:
            best_score = total_est
            best_local_idx = li

    # Pair-lookahead: for the top-3 candidates, also try picking a second item
    # that fills the remaining gap well, to break ties or find better combos
    if fn >= 2 and elapsed < _TOTAL_BUDGET * 0.85:
        top3 = np.argsort(-scores_arr)[:min(3, fn)]
        for li in top3:
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            if remaining_cap <= 0:
                continue
            # Find best second item that fits in remaining_cap
            second_mask = np.ones(n, dtype=bool)
            second_mask[feasible_indices[li]] = False
            second_mask &= remaining_weights <= remaining_cap
            if not np.any(second_mask):
                continue
            second_idxs = np.where(second_mask)[0]
            sw = remaining_weights[second_idxs]
            sv = remaining_values[second_idxs]
            if len(sw) > 0:
                # Pick the second item by density
                best_second_local = int(np.argmax(sv / sw))
                best_second = second_idxs[best_second_local]
                pair_remaining_cap = remaining_cap - sw[best_second_local]
                est2 = _greedy_fill_fast(pair_remaining_cap, remaining_weights,
                                         remaining_values, best_second)
                total_pair = item_v + sv[best_second_local] + est2
                if total_pair > best_score:
                    best_score = total_pair
                    best_local_idx = li

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
