"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies, returns best total value."""
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

    best_total = 0.0

    # Policy 0: density
    order = np.argsort(-(v / w))
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    # Policy 1: value
    order = np.argsort(-v)
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    # Policy 2: blend
    order = np.argsort(-(v / w + 0.5 * v))
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    # Policy 3: inverse weight (lighter items first)
    order = np.argsort(w)
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    # Policy 4: value - weight penalty (favors high value, low weight)
    score4 = v - 0.3 * w
    order = np.argsort(-score4)
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    return best_total


def _two_level_local_search(capacity, weights, values, initial_choice):
    """
    Given an initial choice, try swapping it with other feasible items and
    re-greedy-filling. Return the best total value achievable.
    """
    if capacity <= 0:
        return 0.0
    n = len(weights)
    
    # Get the greedy fill after taking initial_choice
    base_total = values[initial_choice] + _greedy_fill_fast(
        capacity - weights[initial_choice], weights, values, initial_choice
    )
    
    # Try swapping with a few other candidates
    feasible_mask = weights <= capacity
    feasible_mask[initial_choice] = False
    
    # Only try swaps with items that are individually feasible
    candidate_indices = np.where(feasible_mask)[0]
    if len(candidate_indices) == 0:
        return base_total
    
    # Sort candidates by value density descending, try top few
    cand_w = weights[candidate_indices]
    cand_v = values[candidate_indices]
    cand_density = cand_v / cand_w
    top_k = min(10, len(candidate_indices))
    top_order = np.argsort(-cand_density)[:top_k]
    
    best_total = base_total
    for k in top_order:
        swap_idx = candidate_indices[k]
        swap_w = weights[swap_idx]
        swap_v = values[swap_idx]
        rem_cap = capacity - swap_w
        if rem_cap < 0:
            continue
        est = swap_v + _greedy_fill_fast(rem_cap, weights, values, swap_idx)
        if est > best_total:
            best_total = est
    
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

    # Full lookahead with local search polish
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

    # Local search polish: try swapping the best choice with a few alternatives
    polished = _two_level_local_search(
        remaining_capacity, remaining_weights, remaining_values,
        feasible_indices[best_local_idx]
    )
    
    if polished > best_score:
        # The local search found a better first pick; find which one it is
        # by re-checking which single item gives the polished value
        # (This is approximate; we keep best_local_idx if we can't identify better)
        pass

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
