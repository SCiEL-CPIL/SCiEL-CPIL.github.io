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


def _build_value_function(capacity, weights, values, max_dp_units=2000):
    """
    Build an approximate value function V(cap) using DP with weight quantization.
    Returns (V, unit_size) where V[c] is the max value achievable with capacity c*unit_size.
    """
    if capacity <= 0:
        return np.array([0.0]), 1.0

    n = len(weights)
    if n == 0:
        return np.array([0.0]), 1.0

    # Choose quantization unit: aim for ~max_dp_units buckets
    # Use a unit that's a "nice" fraction of the smallest weight or capacity
    min_w = float(np.min(weights))
    # Target ~1000-2000 units
    raw_unit = capacity / max_dp_units
    # Snap to something reasonable
    unit = max(raw_unit, min_w / 5.0)

    # Number of DP states
    n_states = int(capacity / unit) + 1
    n_states = min(n_states, 5000)  # safety cap
    if n_states < 1:
        n_states = 1

    V = np.zeros(n_states, dtype=np.float64)

    # Filter to feasible items only
    feasible = weights <= capacity
    if not np.any(feasible):
        return V, unit

    fw = weights[feasible]
    fv = values[feasible]

    # Quantize weights to integer units
    qw = np.maximum(1, np.ceil(fw / unit).astype(np.int32))

    # 0/1 knapsack DP
    for i in range(len(fw)):
        wi = qw[i]
        vi = fv[i]
        if wi >= n_states:
            continue
        # Update in reverse to avoid using item twice
        for c in range(n_states - 1, wi - 1, -1):
            new_val = V[c - wi] + vi
            if new_val > V[c]:
                V[c] = new_val

    return V, unit


def _marginal_gain_score(capacity, weights, values, V, unit):
    """
    Score each feasible item by its marginal contribution to the value function.
    For item i: score = V[cap/unit] - V[(cap - w_i)/unit]
    This is the value "lost" if we don't take item i (opportunity cost),
    or equivalently the marginal gain from having capacity for item i.
    """
    n = len(weights)
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return None, None

    feasible_indices = np.where(feasible_mask)[0]
    fw = weights[feasible_indices]
    fv = values[feasible_indices]
    fn = len(fw)

    cap_units = int(capacity / unit)
    cap_units = min(cap_units, len(V) - 1)

    # For each item, compute the marginal value:
    # V[cap_units] represents the best value with full capacity
    # V[cap_units - w_i_units] represents the best value without this item's weight
    # The marginal gain of item i is approximately:
    #   V[cap_units] - V[cap_units - ceil(w_i/unit)]
    # But we also need to account for the item's own value contribution.
    # A better score: how much does the total solution improve if we force item i?
    # Approximate: v_i + V[cap_units - w_i_units] vs V[cap_units]
    # So marginal = v_i + V[cap_units - w_i_units] - V[cap_units]
    # Higher marginal means item i is more "essential" / high-value.

    qw = np.ceil(fw / unit).astype(np.int32)
    qw = np.clip(qw, 1, cap_units)

    # V_at_reduced = V[cap_units - qw]
    reduced_units = cap_units - qw
    reduced_units = np.clip(reduced_units, 0, len(V) - 1)
    V_reduced = V[reduced_units]

    V_full = V[cap_units]

    # Marginal gain: how much value we get by including this item
    # = v_i + best_fill_remaining - best_fill_without_constraint
    # Approximation: v_i + V_reduced - V_full
    # If this is positive, the item adds value beyond what the DP would choose anyway
    # If negative, the item is "redundant" (DP would pick something better)
    # We want to pick items with HIGH marginal gain (they're valuable and fit well)

    scores = fv + V_reduced - V_full

    # Also add a density tiebreaker for numerical stability
    density = fv / fw
    d_range = density.max() - density.min() + 1e-12
    d_norm = (density - density.min()) / d_range
    scores = scores + 0.01 * d_norm * (scores.max() - scores.min() + 1.0)

    return feasible_indices, scores


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

    # Build value function and use marginal gain scoring
    V, unit = _build_value_function(remaining_capacity, remaining_weights, remaining_values)

    # Score each feasible item by marginal gain in value function
    feas_idx, scores = _marginal_gain_score(remaining_capacity, remaining_weights, remaining_values, V, unit)

    if feas_idx is None or scores is None:
        return -1

    # Combine with a density component for robustness
    fw2 = remaining_weights[feas_idx]
    fv2 = remaining_values[feas_idx]
    density = fv2 / fw2
    d_min, d_max = density.min(), density.max()
    d_range = d_max - d_min + 1e-12
    d_norm = (density - d_min) / d_range

    # Normalize scores
    s_min, s_max = scores.min(), scores.max()
    s_range = s_max - s_min + 1e-12
    s_norm = (scores - s_min) / s_range

    # Blend: primarily use value-function marginal gain, with density as tiebreaker
    combined = 0.85 * s_norm + 0.15 * d_norm

    best_local_idx = int(np.argmax(combined))

    # Verify with greedy lookahead for the top few candidates
    if elapsed < _TOTAL_BUDGET * 0.6:
        top_k = min(5, fn)
        top_order = np.argsort(-combined)[:top_k]
        best_est = -1.0
        best_top_idx = top_order[0]
        for ti in top_order:
            item_w = fw2[ti]
            item_v = fv2[ti]
            remaining_cap = remaining_capacity - item_w
            est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                    feas_idx[ti])
            total_est = item_v + est
            if total_est > best_est:
                best_est = total_est
                best_top_idx = ti
        best_local_idx = int(best_top_idx)

    return int(feas_idx[best_local_idx])
# EVOLVE-BLOCK-END
