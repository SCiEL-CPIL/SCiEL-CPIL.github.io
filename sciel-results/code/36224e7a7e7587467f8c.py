"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _evaluate_objective(result, demands, capacities, open_cost, assign_cost):
    """Compute objective value. Returns inf if invalid."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    # Sum of open costs for facilities with at least one customer
    used = np.zeros(n_fac, dtype=bool)
    for f in result:
        used[f] = True
    open_cost_sum = np.sum(open_cost[used])
    # Sum of assignment costs
    assign_cost_sum = np.sum(assign_cost[np.arange(len(result)), result])
    return open_cost_sum + assign_cost_sum


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=5000):
    """Improve assignment via local search: try moving each customer to a better facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    current_obj = _evaluate_objective(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_obj):
        return result
    
    improved = True
    iteration = 0
    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        # Try moving each customer
        for i in range(n_cust):
            old_f = result[i]
            if old_f < 0 or old_f >= n_fac:
                continue
            # Try all other facilities
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                # Check capacity
                if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                    continue
                
                # Compute delta in objective
                # Remove i from old_f, add i to new_f
                old_open = open_cost[old_f] if (load[old_f] - demands[i] > 0) else 0.0
                new_open = open_cost[new_f] if (load[new_f] > 0) else open_cost[new_f]
                
                # If old_f becomes empty, we save its open cost
                # If new_f was empty, we incur its open cost
                delta_open = 0.0
                if load[old_f] - demands[i] <= 1e-9:
                    delta_open -= open_cost[old_f]
                if load[new_f] <= 1e-9:
                    delta_open += open_cost[new_f]
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                delta = delta_open + delta_assign
                
                if delta < -1e-9:
                    # Apply the move
                    load[old_f] -= demands[i]
                    load[new_f] += demands[i]
                    result[i] = new_f
                    current_obj += delta
                    improved = True
                    break
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    # Largest demand first, cheapest feasible facility.
    result = np.zeros(len(demands), dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    
    # Local search refinement
    result = _local_search(result, demands, capacities, open_cost, assign_cost)
    
    return result
# EVOLVE-BLOCK-END
