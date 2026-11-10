"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill


def _greedy_complete(capacity, weights, values):
    """Complete greedily by value density; return total value."""
    if len(values) == 0 or capacity <= 0:
        return 0.0
    total = 0.0
    cap = capacity
    w = weights.copy()
    v = values.copy()
    n = len(w)
    for _ in range(n):
        if cap <= 0:
            break
        feasible = w <= cap
        if not np.any(feasible):
            break
        idxs = np.where(feasible)[0]
        if len(idxs) == 0:
            break
        densities = v[idxs] / w[idxs]
        best = idxs[np.argmax(densities)]
        total += v[best]
        cap -= w[best]
        w = np.delete(w, best)
        v = np.delete(v, best)
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item using 1-step lookahead greedy with swap awareness."""
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1

    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Compute densities for feasible items
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    densities = fv / fw

    # Sort by density descending, take top candidates for look-ahead
    # Limit candidates to keep runtime bounded
    max_candidates = min(len(feasible_indices), 30)
    order = np.argsort(-densities)
    top_candidates = order[:max_candidates]

    best_simulated_total = -1.0
    best_candidate_idx = int(feasible_indices[top_candidates[0]])

    for ci in top_candidates:
        # Simulate: pick this item, then greedily fill the rest
        picked_weight = fw[ci]
        picked_value = fv[ci]
        remaining_cap = remaining_capacity - picked_weight

        # Build arrays without this item
        # Use a mask approach for speed
        other_mask = np.ones(len(feasible_indices), dtype=bool)
        other_mask[ci] = False
        other_idx = feasible_indices[other_mask]
        other_w = remaining_weights[other_idx]
        other_v = remaining_values[other_idx]

        sim_total = picked_value + _greedy_complete(remaining_cap, other_w, other_v)

        if sim_total > best_simulated_total:
            best_simulated_total = sim_total
            best_candidate_idx = int(feasible_indices[ci])

    # Deadline check before returning
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -1

    return best_candidate_idx
# EVOLVE-BLOCK-END
