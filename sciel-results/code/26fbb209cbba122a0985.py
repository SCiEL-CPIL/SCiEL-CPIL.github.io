import numpy as np
import time

_START = time.time()
# Safety margin: stop heavy work if we are within this many seconds of the limit.
_DEADLINE = _START + 150.0
_MARGIN = 5.0


def _lp_relax_value(weights, values, order, cap):
    """LP-relaxation value of the knapsack over (weights, values) with capacity cap.
    `order` is the index order sorted by density descending (precomputed).
    Returns the relaxed (fractional) optimal value. O(n)."""
    if cap <= 0:
        return 0.0
    total = 0.0
    remaining = cap
    for idx in order:
        w = weights[idx]
        if w <= 0:
            # zero-weight positive value always fits
            total += values[idx]
            continue
        if w <= remaining:
            remaining -= w
            total += values[idx]
        else:
            # fractional take
            if remaining > 0:
                total += values[idx] * (remaining / w)
            break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    cap = float(remaining_capacity)

    # Precompute density order once (descending). Zero-weight positive items first.
    densities = np.full(n, -np.inf, dtype=float)
    nz = weights > 0
    densities[nz] = values[nz] / weights[nz]
    # zero weight: infinite density if value>0 else -inf
    zw = weights == 0
    densities[zw & (values > 0)] = np.inf
    order = np.argsort(-densities, kind='stable')

    best_val = -np.inf
    best_idx = -1
    best_density = -np.inf

    # Deadline guard: if we are close to the limit, fall back to a single greedy pick.
    if time.time() > _DEADLINE - _MARGIN:
        # quick greedy: first feasible item in density order
        for idx in order:
            if weights[idx] <= cap:
                return int(idx)
        return -1

    for i in range(n):
        w_i = weights[i]
        if w_i > cap:
            continue
        sim_cap = cap - w_i
        if sim_cap < 0:
            continue
        # LP estimate of the best we can do with the rest
        est = values[i] + _lp_relax_value(weights, values, order, sim_cap)
        d_i = densities[i]
        # tiny tie-break by density to prefer efficient items on ties
        if est > best_val + 1e-12 or (abs(est - best_val) <= 1e-12 and d_i > best_density):
            best_val = est
            best_idx = i
            best_density = d_i

    return int(best_idx) if best_idx != -1 else -1
