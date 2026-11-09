"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _greedy_total(cap, weights, values, order, skip_idx=-1):
    """Simulate greedy fill following density order, skipping skip_idx. Return total value."""
    total_val = 0.0
    for i in order:
        if i == skip_idx:
            continue
        w = weights[i]
        if w <= cap:
            cap -= w
            total_val += values[i]
        if cap <= 0:
            break
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check deadline
    if not _elapsed_ok():
        # Fallback to simple greedy
        densities = remaining_values / remaining_weights
        return int(np.argmax(densities))
    
    # Compute density order (highest first)
    densities = remaining_values / remaining_weights
    order = np.argsort(-densities)  # descending density
    
    # Find all items that fit in remaining capacity
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # For each feasible candidate, simulate the rest of the greedy fill
    # and pick the one that maximizes total value
    best_idx = -1
    best_total = -1.0
    
    # Limit lookahead to top candidates by density for speed
    # Sort feasible by density descending
    feasible_sorted = feasible_indices[np.argsort(-densities[feasible_indices])]
    
    # Try up to top 20 candidates by density (or all if fewer)
    max_candidates = min(len(feasible_sorted), 20)
    
    for skip_idx in feasible_sorted[:max_candidates]:
        total = _greedy_total(remaining_capacity, remaining_weights, remaining_values, order, skip_idx)
        # Add the value of the item we're considering
        total += remaining_values[skip_idx]
        if total > best_total:
            best_total = total
            best_idx = int(skip_idx)
    
    if best_idx == -1:
        # Fallback
        return int(np.argmax(densities))
    
    return best_idx
# EVOLVE-BLOCK-END
