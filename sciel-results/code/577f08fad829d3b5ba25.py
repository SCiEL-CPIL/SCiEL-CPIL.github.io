"""Bounded lookahead 0/1 knapsack heuristic with multi-policy greedy completion."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 115.0


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _greedy_complete(w, v, cap, policy='density'):
    """Simulate greedy completion. Returns total added value."""
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0
    if policy == 'density':
        order = np.argsort(-v / np.maximum(w, 1e-12))
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        order = np.argsort(-(v / np.maximum(w, 1e-12) + 0.15 * v))
    else:
        order = np.argsort(-v / np.maximum(w, 1e-12))

    rem = float(cap)
    total = 0.0
    for i in order:
        wi = float(w[i])
        if wi <= rem:
            rem -= wi
            total += float(v[i])
            if rem <= 0:
                break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add using lookahead greedy completion.
    For each candidate item, simulate adding it plus a density-greedy fill
    of the remaining capacity, and choose the item that maximizes total
    estimated value.
    """
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    n = len(w)

    if n == 0 or cap <= 0:
        return -1

    # Only consider feasible items (weight <= capacity)
    feasible = w <= cap
    if not np.any(feasible):
        return -1

    feasible_indices = np.where(feasible)[0]
    fw = w[feasible]
    fv = v[feasible]
    n_feasible = len(fw)

    # Check time budget
    if _remaining_budget() < 0.5:
        # Fallback to simple density
        density = fv / np.maximum(fw, 1e-12)
        best_local = int(np.argmax(density))
        return int(feasible_indices[best_local])

    # Evaluate each candidate item with lookahead
    # For small n, evaluate all; for larger n, sample top candidates by density
    if n_feasible <= 30:
        candidates = feasible_indices
    else:
        # Take top 30 by density as candidates for lookahead
        density = fv / np.maximum(fw, 1e-12)
        top_k = min(30, n_feasible)
        top_local = np.argsort(-density)[:top_k]
        candidates = feasible_indices[top_local]

    best_score = -1.0
    best_candidate = None
    best_policy = 'density'

    for ci in candidates:
        wi = float(w[ci])
        vi = float(v[ci])
        new_cap = cap - wi
        if new_cap < 0:
            continue

        # Create remaining arrays after taking this item
        mask = np.ones(n, dtype=bool)
        mask[ci] = False
        rem_w = w[mask]
        rem_v = v[mask]

        # Try multiple policies for completion
        for policy in ('density', 'value', 'blend'):
            completion_value = _greedy_complete(rem_w, rem_v, new_cap, policy)
            total_value = vi + completion_value
            if total_value > best_score:
                best_score = total_value
                best_candidate = ci
                best_policy = policy

    if best_candidate is None:
        # Fallback to highest density feasible item
        density = fv / np.maximum(fw, 1e-12)
        best_local = int(np.argmax(density))
        return int(feasible_indices[best_local])

    return int(best_candidate)
# EVOLVE-BLOCK-END
