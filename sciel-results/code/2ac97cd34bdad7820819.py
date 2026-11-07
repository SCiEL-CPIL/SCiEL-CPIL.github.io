"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _greedy_init(demands, capacities, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n = len(demands)
    result = np.zeros(n, dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result, load


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return np.inf
        load[f] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            return np.inf
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    total += np.sum(assign_cost[np.arange(len(result)), result])
    return total


def _try_flow(demands, capacities, open_cost, assign_cost, deadline):
    """
    Use min-cost flow via LP relaxation:
    Create a bipartite graph: customers -> facility-slots.
    Each facility f has slots = capacities[f] // min_demand (rough).
    Actually, use a different approach: formulate as assignment with capacity constraints
    by duplicating facilities.
    
    Simpler: use scipy linear_sum_assignment on a cost matrix where we handle
    capacities by penalizing over-capacity, then repair.
    
    Alternative structural approach: 
    1. Start with greedy
    2. Local search: try moving one customer to a different facility if it improves cost
    3. Try swapping customers between facilities
    """
    n = len(demands)
    n_fac = len(capacities)
    result, load = _greedy_init(demands, capacities, assign_cost)
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(current_cost):
        return result, current_cost
    
    deadline = deadline - 0.5  # safety margin
    
    # Local search: single moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try moving each customer to a different facility
        for i in range(n):
            if time.time() > deadline:
                break
            best_gain = 0
            best_f = -1
            current_f = result[i]
            for f in range(n_fac):
                if f == current_f:
                    continue
                # Check if f has capacity
                if load[f] + demands[i] > capacities[f] + 1e-9:
                    continue
                # Compute cost change
                # Remove i from current_f: might close current_f if it was last customer
                old_load_cf = load[current_f]
                new_load_cf = old_load_cf - demands[i]
                old_open = open_cost[current_f] if old_load_cf > 0 else 0
                new_open = open_cost[current_f] if new_load_cf > 0 else 0
                
                old_load_f = load[f]
                new_load_f = old_load_f + demands[i]
                old_open_f = open_cost[f] if old_load_f > 0 else 0
                new_open_f = open_cost[f] if new_load_f > 0 else 0
                
                old_assign = assign_cost[i, current_f]
                new_assign = assign_cost[i, f]
                
                gain = (old_assign + old_open + old_open_f) - (new_assign + new_open + new_open_f)
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
            
            if best_f >= 0 and best_gain > 1e-9:
                result[i] = best_f
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
                current_cost -= best_gain
                improved = True
    
    # Try pairwise swaps
    if time.time() < deadline:
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                # Check capacity after swap
                if load[fi] - demands[i] + demands[j] > capacities[fi] + 1e-9:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj] + 1e-9:
                    continue
                
                # Compute cost change
                old_load_fi = load[fi]
                old_load_fj = load[fj]
                new_load_fi = old_load_fi - demands[i] + demands[j]
                new_load_fj = old_load_fj - demands[j] + demands[i]
                
                old_open_fi = open_cost[fi] if old_load_fi > 0 else 0
                new_open_fi = open_cost[fi] if new_load_fi > 0 else 0
                old_open_fj = open_cost[fj] if old_load_fj > 0 else 0
                new_open_fj = open_cost[fj] if new_load_fj > 0 else 0
                
                old_assign = assign_cost[i, fi] + assign_cost[j, fj]
                new_assign = assign_cost[i, fj] + assign_cost[j, fi]
                
                gain = (old_assign + old_open_fi + old_open_fj) - (new_assign + new_open_fi + new_open_fj)
                if gain > 1e-9:
                    result[i] = fj
                    result[j] = fi
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    current_cost -= gain
    
    return result, current_cost


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin from 150s
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    result, cost = _try_flow(demands, capacities, open_cost, assign_cost, deadline)
    
    # Final validation
    n_fac = len(capacities)
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(result):
        load[f] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            # Fallback to greedy
            result, _ = _greedy_init(demands, capacities, assign_cost)
            break
    
    return result.astype(int)
# EVOLVE-BLOCK-END
