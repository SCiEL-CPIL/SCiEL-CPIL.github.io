"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _true_obj(result, demands, capacities, open_cost, assign_cost):
    """Compute the objective value for a given assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    # Check capacity constraints
    if np.any(load > capacities):
        return float('inf')
    # Calculate open cost + assignment cost
    used = np.zeros(n_facilities, dtype=bool)
    for i in range(n_customers):
        used[result[i]] = True
    open_cost_sum = np.sum(open_cost[used])
    assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
    return open_cost_sum + assign_cost_sum


def _greedy_seed(demands, capacities, open_cost, assign_cost):
    """Greedy seed: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, time_budget=100.0):
    """Local search with single moves and swap moves, time-budgeted."""
    start_time = time.time()
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = result.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    current_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if not np.isfinite(current_obj):
        return result
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        # Single move search
        for i in np.random.permutation(n_customers):
            if (time.time() - start_time) >= time_budget:
                break
            cur_f = result[i]
            best_f = -1
            best_delta = 0
            for f in range(n_facilities):
                if f == cur_f:
                    continue
                # Check if moving i from cur_f to f is feasible
                if load[f] + demands[i] <= capacities[f]:
                    # Calculate delta
                    delta = assign_cost[i, f] - assign_cost[i, cur_f]
                    # Open cost change: if cur_f becomes empty, save open_cost[cur_f]
                    # if f was empty, add open_cost[f]
                    if load[cur_f] == demands[i]:
                        delta -= open_cost[cur_f]
                    if load[f] == 0:
                        delta += open_cost[f]
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f
            if best_f != -1:
                load[cur_f] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                current_obj += best_delta
                improved = True
        # Swap move search
        if (time.time() - start_time) < time_budget:
            for _ in range(n_customers):
                if (time.time() - start_time) >= time_budget:
                    break
                i1 = np.random.randint(n_customers)
                i2 = np.random.randint(n_customers)
                if i1 == i2:
                    continue
                f1 = result[i1]
                f2 = result[i2]
                if f1 == f2:
                    continue
                # Check feasibility of swap
                if load[f1] - demands[i1] + demands[i2] <= capacities[f1] and \
                   load[f2] - demands[i2] + demands[i1] <= capacities[f2]:
                    delta = (assign_cost[i1, f2] + assign_cost[i2, f1]) - \
                            (assign_cost[i1, f1] + assign_cost[i2, f2])
                    # Open cost changes
                    # f1: load changes by -demands[i1]+demands[i2]
                    # f2: load changes by -demands[i2]+demands[i1]
                    old_load_f1 = load[f1]
                    old_load_f2 = load[f2]
                    new_load_f1 = old_load_f1 - demands[i1] + demands[i2]
                    new_load_f2 = old_load_f2 - demands[i2] + demands[i1]
                    # Open cost delta
                    open_delta = 0
                    if old_load_f1 > 0 and new_load_f1 == 0:
                        open_delta -= open_cost[f1]
                    if old_load_f1 == 0 and new_load_f1 > 0:
                        open_delta += open_cost[f1]
                    if old_load_f2 > 0 and new_load_f2 == 0:
                        open_delta -= open_cost[f2]
                    if old_load_f2 == 0 and new_load_f2 > 0:
                        open_delta += open_cost[f2]
                    total_delta = delta + open_delta
                    if total_delta < -1e-9:
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        result[i1] = f2
                        result[i2] = f1
                        current_obj += total_delta
                        improved = True
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Main entry point: greedy seed + local search with time budget."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate greedy seed
    result = _greedy_seed(demands, capacities, open_cost, assign_cost)
    
    # Local search with remaining time budget
    elapsed = time.time() - start_time
    time_budget = max(0.1, deadline - elapsed)
    result = _local_search(result, demands, capacities, open_cost, assign_cost, time_budget)
    
    # Ensure valid result
    return result
# EVOLVE-BLOCK-END
