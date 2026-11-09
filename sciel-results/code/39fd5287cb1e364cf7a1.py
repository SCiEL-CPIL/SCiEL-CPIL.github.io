"""Rewritten 0/1 knapsack heuristic: 1-step lookahead greedy with full-density completion simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 10.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def _simulate_completion(capacity, weights, values, order, exclude):
    """Fill greedily following `order`, skipping `exclude`, returning total value."""
    cap = capacity
    total = 0.0
    for idx in order:
        if idx == exclude:
            continue
        w = weights[idx]
        if w <= cap:
            cap -= w
            total += values[idx]
            if cap <= 1e-12:
                break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)

    # Feasible candidates: weight <= remaining capacity.
    feasible = np.where(weights <= remaining_capacity + 1e-12)[0]
    if len(feasible) == 0:
        return -1

    # Deadline guard: if we're running out of time, fall back to a fast density pick.
    if not _elapsed_ok():
        dens = values[feasible] / np.maximum(weights[feasible], 1e-12)
        best_local = int(feasible[np.argmax(dens)])
        return best_local

    # Order the *other* items by value density (descending) for the completion simulation.
    # We build a global density order over all remaining items; for each candidate we
    # simulate the fill using that order, skipping the candidate.
    dens_all = values / np.maximum(weights, 1e-12)
    order = np.argsort(-dens_all)  # descending density

    best_val = -1.0
    best_idx = -1
    for c in feasible:
        # Value if we take candidate c now, then complete greedily by density.
        val_after = values[c] + _simulate_completion(
            remaining_capacity - weights[c], weights, values, order, exclude=c
        )
        if val_after > best_val:
            best_val = val_after
            best_idx = int(c)

    if best_idx < 0:
        # Should not happen; fall back to density.
        dens = values[feasible] / np.maximum(weights[feasible], 1e-12)
        best_idx = int(feasible[np.argmax(dens)])

    return best_idx
# EVOLVE-BLOCK-END
