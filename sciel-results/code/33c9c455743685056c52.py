"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_total_cost(result, assign_cost, open_cost):
    """Compute total cost: open costs for used facilities + assignment costs."""
    n_cust = len(result)
    n_fac = len(open_cost)
    used = np.zeros(n_fac, dtype=bool)
    for i in range(n_cust):
        used[result[i]] = True
    total = np.sum(open_cost[used])
    total += np.sum(assign_cost[np.arange(n_cust), result])
    return total


def _greedy_assign(demands, capacities, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    for i in np.argsort(-demands):
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    
    return result.astype(int)


def _local_search(result, demands, capacities, assign_cost, open_cost, deadline):
    """Local search: try moving each customer to a better facility."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_total_cost(result, assign_cost, open_cost)
    improved = True
    
    # Precompute sorted facility order for each customer (by assignment cost)
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_cust):
            if time.time() > deadline:
                break
            current_f = result[i]
            # Try facilities in order of assignment cost
            for f in sorted_facs[i]:
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Compute delta
                delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Open cost delta
                count_current = np.sum(result == current_f)
                count_f = np.sum(result == f)
                
                delta_open = 0.0
                if count_current == 1:
                    delta_open -= open_cost[current_f]
                if count_f == 0:
                    delta_open += open_cost[f]
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    # Accept improvement
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    best_cost += total_delta
                    improved = True
                    break
    
    return result, best_cost


def _simulated_annealing(result, demands, capacities, assign_cost, open_cost, deadline):
    """Simulated annealing to escape local minima."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_result = result.copy()
    best_cost = _compute_total_cost(result, assign_cost, open_cost)
    current_cost = best_cost
    
    T = 500.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.9995
    
    # Precompute sorted facility order
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    while T > T_min and time.time() < deadline:
        # Pick a random customer
        i = np.random.randint(n_cust)
        current_f = result[i]
        
        # Pick a random feasible alternative facility
        feasible_facs = []
        for f in np.random.permutation(n_fac):
            if f != current_f and load[f] + demands[i] <= capacities[f]:
                feasible_facs.append(f)
        
        if not feasible_facs:
            T *= cooling_rate
            continue
        
        f = feasible_facs[np.random.randint(len(feasible_facs))]
        
        # Compute delta
        delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
        count_current = np.sum(result == current_f)
        count_f = np.sum(result == f)
        
        delta_open = 0.0
        if count_current == 1:
            delta_open -= open_cost[current_f]
        if count_f == 0:
            delta_open += open_cost[f]
        
        total_delta = delta_assign + delta_open
        
        if total_delta < 0 or np.random.random() < np.exp(-total_delta / T):
            # Accept
            load[current_f] -= demands[i]
            load[f] += demands[i]
            result[i] = f
            current_cost += total_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
        
        T *= cooling_rate
    
    return best_result, best_cost


def _pairwise_swap(result, demands, capacities, assign_cost, open_cost, deadline):
    """Try pairwise swaps between customers at different facilities."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_total_cost(result, assign_cost, open_cost)
    improved = True
    
    # Group customers by facility
    fac_to_custs = {f: [] for f in range(n_fac)}
    for i in range(n_cust):
        fac_to_custs[result[i]].append(i)
    
    while improved and time.time() < deadline:
        improved = False
        for f1 in range(n_fac):
            if time.time() > deadline:
                break
            for f2 in range(f1 + 1, n_fac):
                if time.time() > deadline:
                    break
                custs1 = list(fac_to_custs[f1])
                custs2 = list(fac_to_custs[f2])
                if not custs1 or not custs2:
                    continue
                for i in custs1:
                    if time.time() > deadline:
                        break
                    for j in custs2:
                        # Check capacity feasibility after swap
                        if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-6:
                            continue
                        if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-6:
                            continue
                        
                        # Compute delta
                        delta = (assign_cost[i, f2] + assign_cost[j, f1]) - \
                                (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        # Open cost delta: only if a facility becomes empty
                        count_f1 = len(fac_to_custs[f1])
                        count_f2 = len(fac_to_custs[f2])
                        delta_open = 0.0
                        if count_f1 == 1:
                            delta_open -= open_cost[f1]
                        if count_f2 == 1:
                            delta_open -= open_cost[f2]
                        
                        total_delta = delta + delta_open
                        if total_delta < -1e-9:
                            # Accept swap
                            load[f1] += demands[j] - demands[i]
                            load[f2] += demands[i] - demands[j]
                            result[i] = f2
                            result[j] = f1
                            best_cost += total_delta
                            improved = True
                            # Update customer groups
                            fac_to_custs[f1].remove(i)
                            fac_to_custs[f2].remove(j)
                            fac_to_custs[f2].append(i)
                            fac_to_custs[f1].append(j)
                            break
                    else:
                        continue
                    break
                if improved and time.time() > deadline:
                    break
    
    return result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each of 100 customers to one of 20 facilities for capacitated facility location.
    Uses greedy initialization, local search, simulated annealing, pairwise swaps, and final polishing.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Phase 1: Greedy initialization
    result = _greedy_assign(demands, capacities, assign_cost)
    
    # Phase 2: Local search
    result, cost = _local_search(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Phase 3: Simulated annealing (if time remains)
    if time.time() < deadline - 10.0:
        result, sa_cost = _simulated_annealing(result, demands, capacities, assign_cost, open_cost, deadline - 10.0)
        cost = sa_cost
    
    # Phase 4: Pairwise swap local search
    result, cost = _pairwise_swap(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Phase 5: Final polishing local search
    result, cost = _local_search(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Verify feasibility
    load = np.zeros(n_fac)
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            result[i] = 0
        load[result[i]] += demands[i]
    
    if np.any(load > capacities + 1e-6):
        # Fallback to greedy if infeasible
        result = _greedy_assign(demands, capacities, assign_cost)
    
    return result.astype(int)
# EVOLVE-BLOCK-END
