"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_left():
    if _START_TIME is None:
        return True
    # Safety margin: return before 145s of 150s limit
    return (time.time() - _START_TIME) < 145.0


def _greedy_value(capacity, weights, values):
    """Simulate greedy fill by value density. Returns total value."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    # Filter to feasible items (weight <= capacity)
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    w = weights[feasible]
    v = values[feasible]
    # Sort by density descending
    order = np.argsort(-v / w)
    w = w[order]
    v = v[order]
    total = 0.0
    cap = capacity
    for wi, vi in zip(w, v):
        if wi <= cap:
            cap -= wi
            total += vi
        else:
            break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1

    # If only one feasible item, take it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # One-step lookahead: for each feasible candidate, simulate greedy completion
    # after taking that candidate, pick the one with max total value
    best_val = -1.0
    best_idx = int(feasible_indices[0])

    for idx in feasible_indices:
        w_i = remaining_weights[idx]
        v_i = remaining_values[idx]
        new_cap = remaining_capacity - w_i

        # Build arrays for remaining items excluding idx
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        w_rest = remaining_weights[mask]
        v_rest = remaining_values[mask]

        total_val = v_i + _greedy_value(new_cap, w_rest, v_rest)

        if total_val > best_val:
            best_val = total_val
            best_idx = int(idx)

        # Time guard
        if not _time_left():
            break

    return best_idx
# EVOLVE-BLOCK-END
