"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _greedy_value(capacity, weights, values, order):
    """Simulate greedy fill following `order`. Return total value."""
    cap = capacity
    val = 0.0
    for i in order:
        w = weights[i]
        if w <= cap:
            cap -= w
            val += values[i]
        if cap <= 0:
            break
    return val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays for speed
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)

    # If only one item, pick it if feasible
    if n == 1:
        return 0 if weights[0] <= cap else -1

    # Compute value density
    with np.errstate(divide='ignore', invalid='ignore'):
        density = values / weights
    density = np.where(weights > 0, density, np.inf)

    # Primary heuristic: highest density
    primary_order = np.argsort(-density, kind='stable')

    # Check time budget
    if not _elapsed_ok():
        # Just return the best density item
        best = int(primary_order[0])
        return best if weights[best] <= cap else -1

    # 1-step lookahead: for each candidate item, simulate what happens if we
    # pick it first, then greedily fill the rest by density
    # To keep it fast, we limit the number of candidates we evaluate in lookahead
    max_lookahead = min(n, 20)
    
    # Candidates: top items by density that are feasible
    candidates = []
    for idx in primary_order:
        if weights[idx] <= cap:
            candidates.append(int(idx))
            if len(candidates) >= max_lookahead:
                break
    
    if not candidates:
        # No feasible items
        return -1

    if len(candidates) == 1:
        return candidates[0]

    # For each candidate, simulate: pick it, then fill rest greedily by density
    # We need to be careful: after picking candidate c, the remaining items are
    # all except c, and remaining capacity is cap - weights[c]
    
    # Precompute the density-sorted order for the full set
    full_order = [int(i) for i in primary_order]
    
    best_val = -1.0
    best_candidate = candidates[0]
    
    for c in candidates:
        # Simulate: pick c first, then greedily fill rest
        rem_cap = cap - weights[c]
        rem_val = values[c]
        
        if rem_cap <= 0:
            # Can't fit anything else
            if rem_val > best_val:
                best_val = rem_val
                best_candidate = c
            continue
        
        # Greedily fill remaining items (excluding c) by density
        for idx in full_order:
            if idx == c:
                continue
            w = weights[idx]
            if w <= rem_cap:
                rem_cap -= w
                rem_val += values[idx]
            if rem_cap <= 0:
                break
        
        if rem_val > best_val:
            best_val = rem_val
            best_candidate = c
    
    return int(best_candidate)
# EVOLVE-BLOCK-END
