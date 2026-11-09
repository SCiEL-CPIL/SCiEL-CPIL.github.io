"""Capacitated facility location: Multi-start local search with swap moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a local search strategy.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 140s max
    deadline = time.time() + 140.0
    
    # Precompute customer order by demand (largest first) for initialization stability
    # But the core algorithm is vectorized, so we'll use random restarts with different 
    # initializations.
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Initialize a valid solution using a greedy heuristic (fast)
    # This ensures we always have a valid fallback.
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    # Sort customers by demand descending for better greedy packing
    order = np.argsort(-demands)
    for i in order:
        # Find feasible facilities
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            # This shouldn't happen if total capacity >= total demand, 
            # but if it does, we have to pick the one with most remaining capacity
            # or just fail. The problem implies a valid solution exists.
            # For safety, pick the facility with max remaining capacity.
            remaining = capacities - load
            f = np.argmax(remaining)
            if load[f] + demands[i] > capacities[f]:
                # Invalid instance? Return something, though it will be invalid.
                # But the contract says valid instances.
                pass
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            # Among feasible, pick cheapest assignment cost
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]
            
    best_result = result_greedy.copy()
    # Calculate score for greedy
    used_fac = np.unique(best_result)
    score_greedy = np.sum(open_cost[used_fac]) + np.sum(assign_cost[np.arange(n_cust), best_result])
    best_score = score_greedy

    def calc_cost(res):
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def local_search(res, load):
        """Local search with single moves, swap moves, and facility-close moves."""
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            
            # Single customer moves (vectorized)
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                    
                f_old = res[i]
                d_i = demands[i]
                
                # Vectorized feasibility check
                feasible = (load + d_i <= capacities) & (np.arange(n_fac) != f_old)
                if not np.any(feasible):
                    continue
                
                # Compute delta for all feasible facilities
                delta_assign = assign_cost[i, feasible] - assign_cost[i, f_old]
                
                # Open cost changes
                delta_open = np.zeros(np.sum(feasible))
                fac_indices = np.where(feasible)[0]
                
                # If moving i empties f_old, save open_cost[f_old]
                if load[f_old] == d_i:
                    delta_open -= open_cost[f_old]
                
                # If moving i opens a new facility, add open_cost[f_new]
                new_facs = fac_indices[load[fac_indices] == 0]
                if len(new_facs) > 0:
                    # Find positions of new_facs in fac_indices
                    pos_in_fac = np.searchsorted(fac_indices, new_facs)
                    delta_open[pos_in_fac] += open_cost[new_facs]
                
                total_delta = delta_assign + delta_open
                min_idx = np.argmin(total_delta)
                best_delta = total_delta[min_idx]
                
                if best_delta < -1e-9:
                    best_f_new = fac_indices[min_idx]
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    improved = True
            
            # Swap moves: try swapping two customers (sampled for speed)
            n_swaps = min(2000, n_cust * (n_cust - 1) // 2)
            if n_swaps > 0:
                # Sample random pairs
                for _ in range(n_swaps):
                    if time.time() > deadline - 2.0:
                        break
                    
                    i, j = np.random.randint(0, n_cust, size=2)
                    if i == j:
                        continue
                    if i > j:
                        i, j = j, i
                    
                    f_i = res[i]
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                    
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    # Check if swap is feasible
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    # Calculate cost delta
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    # Open cost change: only if a facility becomes empty
                    delta_open = 0.0
                    # f_i loses i, gains j: becomes empty only if load[f_i] == d_i and d_j == 0 (impossible)
                    # f_j loses j, gains i: becomes empty only if load[f_j] == d_j and d_i == 0 (impossible)
                    # So delta_open = 0 for valid swaps
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Perform swap
                        res[i], res[j] = res[j], res[i]
                        load[f_i] -= d_i
                        load[f_i] += d_j
                        load[f_j] -= d_j
                        load[f_j] += d_i
                        improved = True
            
            # Facility-close move: try to close a facility by moving all its customers
            for f_close in range(n_fac):
                if time.time() > deadline - 2.0:
                    break
                
                if load[f_close] == 0:
                    continue
                
                custs = np.where(res == f_close)[0]
                if len(custs) == 0:
                    continue
                
                # Try to move all customers from f_close to other facilities
                # Greedy: for each customer, find best alternative
                new_load = load.copy()
                new_load[f_close] = 0
                feasible_moves = True
                move_costs = 0.0
                new_res = res.copy()
                
                # Sort customers by demand descending for better packing
                sorted_custs = custs[np.argsort(-demands[custs])]
                
                for i in sorted_custs:
                    d_i = demands[i]
                    # Find feasible facilities
                    feasible = np.where(new_load + d_i <= capacities)[0]
                    feasible = feasible[feasible != f_close]  # Exclude f_close
                    if len(feasible) == 0:
                        feasible_moves = False
                        break
                    
                    # Pick cheapest
                    best_f = feasible[np.argmin(assign_cost[i, feasible])]
                    new_res[i] = best_f
                    new_load[best_f] += d_i
                    move_costs += assign_cost[i, best_f] - assign_cost[i, f_close]
                
                if not feasible_moves:
                    continue
                
                # Calculate total delta
                old_assign_cost = np.sum(assign_cost[custs, f_close])
                new_assign_cost = np.sum(assign_cost[custs, new_res[custs]])
                delta_assign = new_assign_cost - old_assign_cost
                
                # Open cost: we close f_close, so save open_cost[f_close]
                # But we might open new facilities
                delta_open = -open_cost[f_close]
                for f in range(n_fac):
                    if f == f_close:
                        continue
                    if load[f] == 0 and new_load[f] > 0:
                        delta_open += open_cost[f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    res = new_res
                    load = new_load
                    improved = True
        
        return res, load
    
    # Multi-start with different alpha penalties (increased count)
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
            
        # 1. Initial Assignment based on effective cost
        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1)
        
        # 2. Repair Capacity Violations
        res = init_res.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
            
        # Repair: move customers from overloaded facilities
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
                
            f = overloaded[0]
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                break
                
            best_cust = -1
            best_g = -1
            min_penalty = np.inf
            
            for i in custs_in_f:
                for g in range(n_fac):
                    if g == f:
                        continue
                    if load[g] + demands[i] <= capacities[g]:
                        penalty = assign_cost[i, g] - assign_cost[i, f]
                        if penalty < min_penalty:
                            min_penalty = penalty
                            best_cust = i
                            best_g = g
            
            if best_cust == -1:
                break
                
            res[best_cust] = best_g
            load[f] -= demands[best_cust]
            load[best_g] += demands[best_cust]
            
        # Verify validity
        if np.all(load <= capacities):
            # Run local search on this solution
            res, load = local_search(res, load)
            score = calc_cost(res)
            if score < best_score:
                best_score = score
                best_result = res.copy()
                
    # Final polish: run additional local search passes on the best solution
    current_res = best_result.copy()
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[current_res[i]] += demands[i]
    
    # Run multiple passes of local search for polishing
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        current_res, current_load = local_search(current_res, current_load)
        score = calc_cost(current_res)
        if score < best_score:
            best_score = score
            best_result = current_res.copy()
        else:
            break

    return best_result
# EVOLVE-BLOCK-END
