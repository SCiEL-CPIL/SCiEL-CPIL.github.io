"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if len(result) != n_cust:
        return np.inf
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            return np.inf
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f]:
            return np.inf
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    for i in range(n_cust):
        total += assign_cost[i, result[i]]
    return total


def _greedy_init(demands, capacities, open_cost, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        assigned = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            f = np.argmax(rem)
            load[f] += demands[i]
            result[i] = f
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=110.0):
    """Local search: move customers to reduce cost while respecting capacity."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try moving each customer to a different facility
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            current_f = result[i]
            current_cost = best_cost
            
            # Calculate cost of moving customer i from current_f to each other facility
            # Change in open cost: if current_f becomes empty, we save open_cost[current_f]
            # if new facility was empty, we add open_cost[new_f]
            
            # Get all facilities sorted by assign_cost for customer i
            for f in np.argsort(assign_cost[i]):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Calculate delta cost
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Check if current facility becomes empty
                if load[current_f] == demands[i]:
                    delta -= open_cost[current_f]
                
                # Check if new facility was empty
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < -1e-9:
                    # Accept the move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    best_cost += delta
                    improved = True
                    break  # Move this customer to the best feasible facility and break
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities using greedy initialization and local search."""
    start_time = time.time()
    
    # Greedy initialization
    result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    # Local search with time limit
    time_limit = max(1.0, 110.0 - (time.time() - start_time))
    result = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit)
    
    return result
# EVOLVE-BLOCK-END
