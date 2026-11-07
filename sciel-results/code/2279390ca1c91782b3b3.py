"""Capacitated facility location: vectorized greedy init + fast local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment. Returns inf if invalid."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return float('inf')
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return float('inf')
    
    open_cost_total = np.sum(open_cost[load > 0])
    assign_cost_total = np.sum(assign_cost[np.arange(n_customers), result])
    
    return open_cost_total + assign_cost_total


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        order = np.argsort(assign_cost[i])
        for f in order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: assign to facility with most remaining capacity
            remaining = capacities - load
            f = int(np.argmax(remaining))
            load[f] += demands[i]
            result[i] = f
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Fast local search: try moving single customers to better facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved:
        improved = False
        if time.time() >= deadline:
            break
        
        for i in range(n_customers):
            if time.time() >= deadline:
                break
            
            current_f = best_result[i]
            current_load_f = load[current_f]
            
            # Find best alternative facility
            best_alt_f = -1
            best_delta = 0.0
            
            # Vectorized cost calculation for all facilities
            # Delta = assign_cost[i, f] - assign_cost[i, current_f] + open_cost changes
            deltas = assign_cost[i, :] - assign_cost[i, current_f]
            
            # Open cost changes:
            # - If current_f becomes empty (load == demands[i]), save open_cost[current_f]
            # - If target f was empty (load[f] == 0), pay open_cost[f]
            
            current_becomes_empty = (current_load_f == demands[i])
            target_was_empty = (load == 0)
            
            if current_becomes_empty:
                deltas -= open_cost[current_f]
            if np.any(target_was_empty):
                deltas[target_was_empty] += open_cost[target_was_empty]
            
            # Exclude current facility
            deltas[current_f] = np.inf
            
            # Check capacity feasibility
            feasible = (load + demands[i] <= capacities + 1e-9)
            deltas[~feasible] = np.inf
            
            # Find best
            if np.any(deltas < best_delta - 1e-9):
                best_alt_f = int(np.argmin(deltas))
                best_delta = deltas[best_alt_f]
            
            if best_alt_f >= 0:
                # Perform the move
                load[current_f] -= demands[i]
                load[best_alt_f] += demands[i]
                best_result[i] = best_alt_f
                best_cost += best_delta
                improved = True
    
    return best_result


def _pair_swap_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Try swapping pairs of customers between facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved:
        improved = False
        if time.time() >= deadline:
            break
        
        for i in range(n_customers):
            if time.time() >= deadline:
                break
            
            fi = best_result[i]
            
            for j in range(i + 1, n_customers):
                if time.time() >= deadline:
                    break
                
                fj = best_result[j]
                if fi == fj:
                    continue
                
                # Check if swap is feasible
                new_load_fi = load[fi] - demands[i] + demands[j]
                new_load_fj = load[fj] - demands[j] + demands[i]
                
                if new_load_fi > capacities[fi] + 1e-9:
                    continue
                if new_load_fj > capacities[fj] + 1e-9:
                    continue
                
                # Compute delta cost
                delta = (-assign_cost[i, fi] - assign_cost[j, fj] + 
                         assign_cost[i, fj] + assign_cost[j, fi])
                
                # Open cost changes
                current_open = 0.0
                new_open = 0.0
                
                if load[fi] > 0:
                    current_open += open_cost[fi]
                if new_load_fi > 0:
                    new_open += open_cost[fi]
                
                if load[fj] > 0:
                    current_open += open_cost[fj]
                if new_load_fj > 0:
                    new_open += open_cost[fj]
                
                delta += new_open - current_open
                
                if delta < -1e-9:
                    # Perform swap
                    load[fi] -= demands[i]
                    load[fj] -= demands[j]
                    load[fi] += demands[j]
                    load[fj] += demands[i]
                    best_result[i] = fj
                    best_result[j] = fi
                    best_cost += delta
                    improved = True
    
    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    """Main entry point: greedy init + local search + pair swaps with deadline guard."""
    start_time = time.time()
    # Leave 5s safety margin before 120s limit
    deadline = start_time + 115.0
    
    # Phase 1: Greedy initialization
    result = _greedy_assign(demands, capacities, open_cost, assign_cost)
    
    # Phase 2: Local search (single moves)
    if time.time() < deadline - 10.0:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, deadline - 5.0)
    
    # Phase 3: Pair swap search
    if time.time() < deadline - 5.0:
        result = _pair_swap_search(result, demands, capacities, open_cost, assign_cost, deadline)
    
    return result
# EVOLVE-BLOCK-END
