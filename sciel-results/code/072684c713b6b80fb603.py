"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 3.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def _simulate_greedy(capacity, weights, values, order_indices, exclude_idx):
    """Simulate greedy fill over order_indices, skipping exclude_idx. Return total value."""
    sim_cap = capacity
    sim_val = 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    if not _elapsed_ok():
        # Fallback: highest density feasible item
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        idxs = np.where(feasible)[0]
        densities = remaining_values[idxs] / remaining_weights[idxs]
        return int(idxs[np.argmax(densities)])

    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    feasible_idx = np.where(feasible_mask)[0]
    f_w = remaining_weights[feasible_idx]
    f_v = remaining_values[feasible_idx]
    m = len(f_w)

    # Density order for the feasible set (local indices)
    densities = f_v / f_w
    order = np.argsort(-densities)  # local indices, best density first

    # 1-step lookahead: for each candidate top item, simulate the rest
    # and pick the candidate whose resulting total value is highest.
    # Limit candidates to top K by density to keep cost low.
    K = min(60, m)
    candidates = order[:K]

    # Precompute the full order list of local indices
    full_order = order.tolist()

    best_val = -1.0
    best_local = -1
    for cand in candidates:
        val = _simulate_greedy(remaining_capacity, f_w, f_v, full_order, cand)
        if val > best_val:
            best_val = val
            best_local = cand
        if not _elapsed_ok():
            break

    if best_local < 0:
        # fallback to best density
        best_local = int(candidates[0])

    # Map local index back to index in remaining arrays
    return int(feasible_idx[best_local])
# EVOLVE-BLOCK-END
