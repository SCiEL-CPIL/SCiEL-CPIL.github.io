"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies + light 2-opt polish, returns best total value."""
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

    def _run_policy(order, cap0):
        total = 0.0
        cap = cap0
        chosen = []
        for i in order:
            wi = w[i]
            if wi <= cap:
                total += v[i]
                cap -= wi
                chosen.append(i)
                if cap <= 0:
                    break
        return total, chosen

    def _polish_2opt(total, chosen, cap0, max_passes=2):
        if len(chosen) < 2:
            return total
        chosen_set = set(chosen)
        for _ in range(max_passes):
            improved = False
            # Try swapping one chosen item with one unchosen item
            for ci in range(len(chosen)):
                ci_idx = chosen[ci]
                ci_w = w[ci_idx]
                ci_v = v[ci_idx]
                for ui in range(len(idxs)):
                    ui_idx = idxs[ui]
                    if ui_idx in chosen_set:
                        continue
                    ui_w = w[ui_idx]
                    ui_v = v[ui_idx]
                    # swap: remove ci, add ui
                    new_w = sum(w[chosen[j]] for j in range(len(chosen)) if j != ci) + ui_w
                    if new_w <= cap0 + 1e-12:
                        delta = ui_v - ci_v
                        if delta > 1e-12:
                            # perform swap
                            chosen[ci] = ui_idx
                            chosen_set.remove(ci_idx)
                            chosen_set.add(ui_idx)
                            total += delta
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return total

    best_total = 0.0
    best_chosen = []

    density = v / w

    # Policy 0: density
    order = np.argsort(-density)
    total, chosen = _run_policy(order, capacity)
    total = _polish_2opt(total, chosen, capacity)
    if total > best_total:
        best_total = total
        best_chosen = list(chosen)

    # Policy 1: value
    order = np.argsort(-v)
    total, chosen = _run_policy(order, capacity)
    total = _polish_2opt(total, chosen, capacity)
    if total > best_total:
        best_total = total
        best_chosen = list(chosen)

    # Policy 2: blend (density + 0.5 * value)
    order = np.argsort(-(density + 0.5 * v))
    total, chosen = _run_policy(order, capacity)
    total = _polish_2opt(total, chosen, capacity)
    if total > best_total:
        best_total = total
        best_chosen = list(chosen)

    # Policy 3: small-weight first (to pack more items)
    order = np.argsort(w)
    total, chosen = _run_policy(order, capacity)
    total = _polish_2opt(total, chosen, capacity)
    if total > best_total:
        best_total = total
        best_chosen = list(chosen)

    # Policy 4: snug-fit preference: density * (1 + 0.15 * (w / capacity))
    if capacity > 0:
        scores = density * (1 + 0.15 * (w / capacity))
        order = np.argsort(-scores)
        total, chosen = _run_policy(order, capacity)
        total = _polish_2opt(total, chosen, capacity)
        if total > best_total:
            best_total = total
            best_chosen = list(chosen)

    # Policy 5: weight-balanced: prefer items with weight close to capacity/3
    if capacity > 0:
        target_w = capacity / 3.0
        w_norm = np.abs(w - target_w) / (target_w + 1e-12)
        scores = density * np.exp(-0.5 * w_norm)
        order = np.argsort(-scores)
        total, chosen = _run_policy(order, capacity)
        total = _polish_2opt(total, chosen, capacity)
        if total > best_total:
            best_total = total
            best_chosen = list(chosen)

    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Select next item using lookahead greedy completion with multiple policies."""
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
        # Blend density and value with a snug-fit bonus
        if remaining_capacity > 0:
            snug = 1 + 0.15 * (fw / remaining_capacity)
            scores = density * snug
        else:
            scores = density
        return int(feasible_indices[int(np.argmax(scores))])

    # Lookahead: for each feasible item, estimate total via greedy completion
    # Use a deadline-aware loop: stop lookahead if time is running out
    deadline = _TOTAL_BUDGET - 5.0  # 5s safety margin before 150s kill
    best_score = -1.0
    best_local_idx = 0

    for li in range(fn):
        if time.monotonic() > deadline:
            break
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
