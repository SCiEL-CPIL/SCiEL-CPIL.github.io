"""Improved 0/1 knapsack heuristic with B&B lookahead evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # seconds; safety margin before 150s kill
_START = None
_BB_NODE_BUDGET = 2000  # max nodes per B&B call


def _fractional_knapsack_bound(weights, values, capacity):
    """Upper bound via fractional knapsack (LP relaxation)."""
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    n = len(weights)
    density = values / weights
    order = np.argsort(density)[::-1]
    w_sorted = weights[order]
    v_sorted = values[order]
    remaining_cap = capacity
    total_val = 0.0
    for i in range(n):
        if remaining_cap <= 0:
            break
        if w_sorted[i] <= remaining_cap:
            total_val += v_sorted[i]
            remaining_cap -= w_sorted[i]
        else:
            total_val += v_sorted[i] * (remaining_cap / w_sorted[i])
            remaining_cap = 0
    return total_val


def _branch_and_bound(weights, values, capacity, node_budget=_BB_NODE_BUDGET):
    """
    Branch and bound for 0/1 knapsack. Returns best integer solution value.
    Uses fractional knapsack as upper bound for pruning.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0

    # Sort by density descending for better pruning
    density = values / weights
    order = np.argsort(density)[::-1]
    w_sorted = weights[order].copy()
    v_sorted = values[order].copy()

    best_val = 0.0
    nodes_visited = 0

    # Stack-based DFS: (index, remaining_capacity, current_value)
    stack = [(0, capacity, 0.0)]

    while stack and nodes_visited < node_budget:
        idx, rem_cap, cur_val = stack.pop()
        nodes_visited += 1

        if idx >= n:
            if cur_val > best_val:
                best_val = cur_val
            continue

        # Upper bound: current value + fractional knapsack of remaining items
        if idx < n:
            ub = cur_val + _fractional_knapsack_bound(w_sorted[idx:], v_sorted[idx:], rem_cap)
        else:
            ub = cur_val

        if ub <= best_val:
            continue

        w_i = w_sorted[idx]
        v_i = v_sorted[idx]

        # Option 1: take item idx
        if w_i <= rem_cap:
            new_val = cur_val + v_i
            if new_val > best_val:
                best_val = new_val
            # Push the "take" branch
            stack.append((idx + 1, rem_cap - w_i, new_val))

        # Option 2: skip item idx
        stack.append((idx + 1, rem_cap, cur_val))

    return best_val


def _greedy_fill_value(weights, values, capacity, exclude_idx=None):
    """Compute total value from greedy density fill (0/1, no fractional)."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    if len(w) == 0:
        return 0.0
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    cum_w = np.cumsum(w_sorted)
    cum_v = np.cumsum(v_sorted)
    idx = np.searchsorted(cum_w, capacity, side='right') - 1
    if idx < 0:
        return 0.0
    return float(cum_v[idx])


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item for 0/1 knapsack using B&B lookahead on top-k density candidates.
    """
    global _START
    if _START is None:
        _START = time.time()
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Fallback: pick highest value density feasible item
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        densities = np.where(feasible, remaining_values / remaining_weights, 0)
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    if remaining_capacity <= 0:
        return -1

    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1

    # Only consider feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    feasible_densities = feasible_values / feasible_weights

    # Top-k candidates by density
    k = min(len(feasible_indices), 20)
    top_k_local = np.argsort(feasible_densities)[::-1][:k]
    top_k_global = feasible_indices[top_k_local]

    # For each candidate, compute lookahead: v_i + B&B(remaining after taking i)
    best_score = -1.0
    best_idx = top_k_global[0]

    # Adaptive node budget based on remaining time
    remaining_time = _DEADLINE - (time.time() - _START)
    if remaining_time > 100:
        local_budget = 3000
    elif remaining_time > 50:
        local_budget = 1500
    else:
        local_budget = 500

    for i in top_k_global:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        rem_cap = remaining_capacity - w_i

        # Build arrays excluding item i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_excl = remaining_weights[mask]
        v_excl = remaining_values[mask]

        # Use B&B for tight estimate
        if rem_cap > 0 and len(w_excl) > 0:
            fill_val = _branch_and_bound(w_excl, v_excl, rem_cap, node_budget=local_budget)
        else:
            fill_val = 0.0

        total_val = v_i + fill_val

        # Slight preference for tighter fits
        waste = rem_cap
        score = total_val - 0.0005 * waste

        if score > best_score:
            best_score = score
            best_idx = int(i)

    # Check deadline
    if time.time() - _START > _DEADLINE:
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        densities = np.where(feasible, remaining_values / remaining_weights, 0)
        return int(np.argmax(densities))

    return best_idx
# EVOLVE-BLOCK-END
