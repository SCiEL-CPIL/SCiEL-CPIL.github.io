"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 150s. We want to stay well under.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 138.0  # 12s safety margin


def _lp_order_and_pivot(capacity, weights, values):
    """
    Compute the LP-relaxation optimal order for the fractional knapsack.
    Returns:
        order: array of indices sorted by density descending
        pivot_idx: index in the full array of the break item (fractionally filled), or -1 if none
        lp_value: the LP optimal value
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return np.array([], dtype=int), -1, 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)

    order = np.argsort(-densities, kind='stable')

    lp_value = 0.0
    remaining_cap = capacity
    pivot_idx = -1

    for pos in range(len(order)):
        idx = order[pos]
        w = weights[idx]
        v = values[idx]
        if w <= remaining_cap:
            lp_value += v
            remaining_cap -= w
        else:
            # This is the break item
            if w > 0:
                lp_value += v * (remaining_cap / w)
            pivot_idx = idx
            remaining_cap = 0.0
            break

    return order, pivot_idx, lp_value


def _vectorized_greedy_fill(capacity, weights, values, exclude_local_idx=None):
    """
    Vectorized greedy fill: simulate the density-sorted greedy packing in O(n) numpy ops.
    Returns total value achieved.
    exclude_local_idx: local index to skip (in the weights/values array), or None.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0

    n = len(weights)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)

    order = np.argsort(-densities, kind='stable')

    # Get sorted weights and values
    sw = weights[order]
    sv = values[order]

    # Mask out the excluded item if provided
    if exclude_local_idx is not None:
        # Find position of exclude_local_idx in the sorted order
        pos_in_order = np.where(order == exclude_local_idx)[0]
        if len(pos_in_order) > 0:
            sw = np.delete(sw, pos_in_order[0])
            sv = np.delete(sv, pos_in_order[0])

    if len(sw) == 0:
        return 0.0

    # Cumulative weights to find how many items fit
    cum_w = np.cumsum(sw)
    # Number of items that fully fit
    n_full = np.searchsorted(cum_w, capacity, side='right')

    if n_full == 0:
        return 0.0

    total_value = float(np.sum(sv[:n_full]))
    remaining_cap = capacity - float(cum_w[n_full - 1])

    # Check if there's a fractional next item
    if n_full < len(sw) and remaining_cap > 1e-12:
        next_w = sw[n_full]
        if next_w > 0:
            total_value += sv[n_full] * (remaining_cap / next_w)

    return total_value


def _multi_strategy_greedy(capacity, weights, values, exclude_local_idx=None):
    """
    Run greedy fill under multiple ordering strategies and return the best value.
    Strategies: density (v/w), value, -weight (lightest first), value^0.5/weight.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0

    n = len(weights)

    # Build candidate orderings
    orderings = []

    # 1. Density: v/w
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = np.where(weights > 0, values / weights, 0.0)
    orderings.append(np.argsort(-d1, kind='stable'))

    # 2. Pure value descending
    orderings.append(np.argsort(-values, kind='stable'))

    # 3. Lightest first (min weight)
    orderings.append(np.argsort(weights, kind='stable'))

    # 4. sqrt(value)/weight
    with np.errstate(divide='ignore', invalid='ignore'):
        d4 = np.where(weights > 0, np.sqrt(np.maximum(values, 0.0)) / weights, 0.0)
    orderings.append(np.argsort(-d4, kind='stable'))

    # 5. value / weight^0.7 (slightly favor heavier high-value items)
    with np.errstate(divide='ignore', invalid='ignore'):
        d5 = np.where(weights > 0, values / np.power(weights, 0.7), 0.0)
    orderings.append(np.argsort(-d5, kind='stable'))

    best_val = 0.0
    for order in orderings:
        sw = weights[order]
        sv = values[order]

        if exclude_local_idx is not None:
            pos_in_order = np.where(order == exclude_local_idx)[0]
            if len(pos_in_order) > 0:
                sw = np.delete(sw, pos_in_order[0])
                sv = np.delete(sv, pos_in_order[0])

        if len(sw) == 0:
            continue

        cum_w = np.cumsum(sw)
        n_full = np.searchsorted(cum_w, capacity, side='right')

        if n_full == 0:
            continue

        val = float(np.sum(sv[:n_full]))
        remaining_cap = capacity - float(cum_w[n_full - 1])

        if n_full < len(sw) and remaining_cap > 1e-12:
            next_w = sw[n_full]
            if next_w > 0:
                val += sv[n_full] * (remaining_cap / next_w)

        if val > best_val:
            best_val = val

    return best_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item index (into the remaining arrays) to add, or -1 to stop.

    Strategy: LP-relaxation-guided heuristic with vectorized multi-strategy lookahead.
    1. Compute the fractional knapsack optimal order and identify the pivot (break) item.
    2. Score each feasible item using:
       - LP alignment (structural prior from fractional knapsack)
       - Multi-strategy greedy lookahead (best of 5 orderings)
       - Soft-max combination for smooth scoring
    3. Return the highest-scoring item.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]

    n_feasible = len(feasible_indices)

    # If only one feasible item, pick it
    if n_feasible == 1:
        return int(feasible_indices[0])

    # Compute LP relaxation order and pivot
    order, pivot_idx, lp_value = _lp_order_and_pivot(
        remaining_capacity, weights, values
    )

    # Build a rank map: for each global index, its position in the LP order
    rank_map = np.full(n, -1, dtype=np.int64)
    for pos, idx in enumerate(order):
        rank_map[idx] = pos

    if pivot_idx >= 0:
        pivot_rank = rank_map[pivot_idx]
    else:
        pivot_rank = len(order)  # no pivot, all items fit

    # Precompute LP alignment scores for all feasible items (vectorized)
    lp_alignments = np.zeros(n_feasible, dtype=np.float64)
    for local_i in range(n_feasible):
        global_i = feasible_indices[local_i]
        rank_i = rank_map[global_i]

        if rank_i < 0:
            lp_alignments[local_i] = 0.0
        elif rank_i < pivot_rank:
            lp_alignments[local_i] = 1.0 + (pivot_rank - rank_i) * 0.1
        elif rank_i == pivot_rank:
            lp_alignments[local_i] = 0.5
        else:
            lp_alignments[local_i] = max(0.0, 0.3 - (rank_i - pivot_rank) * 0.05)

    # Also compute density-based score as an auxiliary signal
    with np.errstate(divide='ignore', invalid='ignore'):
        feasible_densities = np.where(feasible_weights > 0, feasible_values / feasible_weights, 0.0)

    # Normalize density to [0, 1] for blending
    max_dens = np.max(feasible_densities) if np.max(feasible_densities) > 0 else 1.0
    norm_densities = feasible_densities / max_dens

    # Multi-strategy lookahead: for each candidate, compute best greedy fill after taking it
    # This is the expensive part - vectorize where possible
    future_values = np.zeros(n_feasible, dtype=np.float64)

    # Check if we have enough time for multi-strategy; otherwise fall back to single strategy
    elapsed = time.time() - _START_TIME
    time_left = _DEADLINE - time.time()
    use_multi = time_left > 5.0 and n_feasible <= 80

    for local_i in range(n_feasible):
        if time.time() > _DEADLINE - 1.0:
            break

        global_i = feasible_indices[local_i]
        w_i = feasible_weights[local_i]
        v_i = feasible_values[local_i]
        cap_after = remaining_capacity - w_i

        if cap_after < 0:
            future_values[local_i] = 0.0
            continue

        if use_multi:
            future_values[local_i] = _multi_strategy_greedy(
                cap_after, feasible_weights, feasible_values, exclude_local_idx=local_i
            )
        else:
            future_values[local_i] = _vectorized_greedy_fill(
                cap_after, feasible_weights, feasible_values, exclude_local_idx=local_i
            )

    # Normalize future_values to [0, 1] for soft-max blending
    max_future = np.max(future_values) if np.max(future_values) > 0 else 1.0
    norm_future = future_values / max_future

    # Soft-max combination of scoring signals:
    # - LP alignment (structural prior)
    # - Normalized future value (lookahead)
    # - Normalized density (direct quality)
    # Use log-sum-exp style soft-max to smoothly combine
    # score = soft_max(lp_align, norm_future, norm_density) with temperature

    # Temperature controls how peaked the combination is
    # Lower temp -> more like max; higher temp -> more like mean
    temp = 2.0

    # Soft-max: exp(s_i / temp) / sum(exp(s_j / temp))
    # We want a score that rewards items high on ALL signals
    # Use: sum of exp(s_i/temp) normalized, then take the log for a smooth "all-good" score
    # Alternatively: weighted geometric mean via log-sum-exp

    # Approach: compute a combined score as weighted sum in log-space
    # log_score = w1*log(lp_align+eps) + w2*log(norm_future+eps) + w3*log(norm_dens+eps)
    # Then exp(log_score) gives a smooth multiplicative score

    eps = 1e-6
    w_lp = 0.4
    w_future = 0.4
    w_dens = 0.2

    # Avoid log(0)
    safe_lp = np.maximum(lp_alignments, eps)
    safe_future = np.maximum(norm_future, eps)
    safe_dens = np.maximum(norm_densities, eps)

    log_scores = (w_lp * np.log(safe_lp) +
                  w_future * np.log(safe_future) +
                  w_dens * np.log(safe_dens))

    # Convert back to linear scale for ranking
    scores = np.exp(log_scores)

    # Add a small LP alignment bonus to break near-ties in favor of LP-optimal items
    # This preserves the structural prior while letting lookahead differentiate
    scores = scores * (1.0 + 0.1 * lp_alignments)

    # Also add the raw value as a tiny tiebreaker
    scores = scores + 1e-4 * feasible_values

    best_local = int(np.argmax(scores))
    return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
