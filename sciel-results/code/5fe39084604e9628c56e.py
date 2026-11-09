"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 2.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def _simulate_greedy(capacity, weights, values, order_indices, exclude_idx):
    """
    Simulates a greedy knapsack solution given a specific ordering of items.
    Returns the total value.
    """
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

    # Compute value density
    densities = remaining_values / remaining_weights

    # Sort indices by density descending
    order = np.argsort(-densities)

    # Consider top candidates for 1-step lookahead
    # We simulate picking each top candidate first, then filling greedily with the rest
    # Limit candidates to top 50 to keep runtime low
    num_candidates = min(50, n)

    best_val = -1
    best_idx = -1

    # Precompute the sorted order excluding each candidate is expensive if done naively.
    # Instead, we can iterate through the sorted order and skip the candidate.

    # To speed up, we can pre-filter: if remaining_capacity is very small, only consider items that fit.
    # But the simulation handles that.

    # We will simulate for each of the top num_candidates items in density order.
    # Note: order[0] is the highest density.
    for c in range(num_candidates):
        if not _elapsed_ok():
            break
        candidate_idx = order[c]
        # Simulate greedy fill starting with candidate_idx, then continuing with order excluding candidate_idx
        # We can do this efficiently by iterating through 'order' and skipping candidate_idx
        sim_cap = remaining_capacity
        sim_val = 0.0

        # Pick candidate first
        w_cand = remaining_weights[candidate_idx]
        if w_cand <= sim_cap:
            sim_cap -= w_cand
            sim_val += remaining_values[candidate_idx]
        else:
            # If candidate doesn't fit, it's not a valid first pick?
            # The function returns an index into remaining arrays. The caller adds it.
            # If it doesn't fit, the caller might reject it or handle it.
            # Usually, the heuristic should only return feasible items?
            # The contract says "picks the next item to add... or -1 to stop".
            # It implies the item should be feasible? Or does the caller check?
            # "remaining (feasible) arrays" suggests the arrays passed in only contain feasible items?
            # "index into the remaining (feasible) arrays" -> Yes, the inputs are already filtered for feasibility.
            # So w_cand <= remaining_capacity is always true.
            pass

        # Continue with the rest of the items in density order, skipping candidate
        for idx in order:
            if idx == candidate_idx:
                continue
            w = remaining_weights[idx]
            if w <= sim_cap:
                sim_cap -= w
                sim_val += remaining_values[idx]
            if sim_cap <= 0:
                break

        if sim_val > best_val:
            best_val = sim_val
            best_idx = candidate_idx

    if best_idx == -1:
        # Fallback to pure density greedy
        return int(order[0])
    else:
        return int(best_idx)
# EVOLVE-BLOCK-END
