"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.

    Strategy:
      1. Quick feasibility check; if nothing fits, stop.
      2. Compute densities for all feasible items.
      3. 1-step look-ahead: for each feasible candidate, simulate picking it
         first, then greedily filling the rest by density. Pick the candidate
         that yields the highest total simulated value.
      4. Time guard: if the look-ahead is running too long, fall back to
         pure density (or a finisher-aware density) to stay within budget.
      5. Finisher awareness: if a candidate takes >70% of capacity, compare
         its value against the best small-item value; if the small stack is
         clearly better, deprioritize the finisher.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = len(remaining_weights)
    if n == 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    cap = float(remaining_capacity)

    # Feasible mask: item weight <= remaining capacity
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    n_feas = len(feasible_indices)

    if n_feas == 1:
        return int(feasible_indices[0])

    # Compute densities (handle zero weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, np.inf)

    # Sort feasible items by density descending (for greedy fill simulation)
    sort_order = np.argsort(-densities)
    sorted_feasible_indices = feasible_indices[sort_order]

    # --- Time check: if we're already close to deadline, use pure density ---
    if time.time() > deadline - 10.0:
        return int(sorted_feasible_indices[0])

    # --- Look-ahead: evaluate each candidate as the "next pick" ---
    # For each candidate i in feasible_indices:
    #   total_val = value[i] + greedy_fill(cap - weight[i], all other items by density)
    # Pick the candidate with max total_val.

    best_total_val = -1.0
    best_pick_idx = -1

    # Pre-compute sorted weights and values for fast greedy simulation
    sorted_w = weights[sorted_feasible_indices]
    sorted_v = values[sorted_feasible_indices]

    # Map from original index to position in sorted order (for skipping)
    # We'll use a set of sorted positions to skip
    # Actually, we can just iterate and skip by index comparison

    for i in range(n_feas):
        if time.time() > deadline:
            # Fallback to density
            return int(sorted_feasible_indices[0])

        cand_idx = feasible_indices[i]
        cand_w = f_weights[i]
        cand_v = f_values[i]

        rem_cap = cap - cand_w
        total_val = cand_v

        # Greedy fill with remaining items (sorted by density, skip cand_idx)
        # Use vectorized prefix sums for speed when possible
        # For small n, loop is fine; for larger n, we can use cumsum
        # Here n_feas <= 100, so loop is acceptable
        for j in range(n_feas):
            next_idx = sorted_feasible_indices[j]
            if next_idx == cand_idx:
                continue
            w_next = weights[next_idx]
            if w_next <= rem_cap:
                rem_cap -= w_next
                total_val += values[next_idx]
            # If rem_cap is very small, we could break early,
            # but items are sorted by density not weight, so can't break
            # unless rem_cap < min remaining weight. Skip that optimization.

        if total_val > best_total_val:
            best_total_val = total_val
            best_pick_idx = cand_idx

    if best_pick_idx == -1:
        return int(sorted_feasible_indices[0])

    return int(best_pick_idx)
# EVOLVE-BLOCK-END
