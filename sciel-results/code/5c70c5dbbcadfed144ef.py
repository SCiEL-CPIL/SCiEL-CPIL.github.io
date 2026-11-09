"""Capacitated facility location solver with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost for a given assignment (vectorized)."""
    load = np.bincount(result, weights=demands, minlength=len(capacities))
    if np.any(load > capacities):
        return float('inf')
    open_facs = np.where(load > 0)[0]
    cost = np.sum(open_cost[open_facs])
    cost += np.sum(assign_cost[np.arange(len(demands)), result])
    return cost


def _greedy_construct(demands, capacities, open_cost, assign_cost, open_penalty=0.0, rng=None):
    """Greedy construction: largest demand first, cheapest feasible facility.
    open_penalty: add to open_cost to discourage opening more facilities."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    # Effective open cost with penalty
    eff_open = open_cost + open_penalty
    
    for i in order:
        # Get facilities sorted by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        placed = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Fallback: find any feasible facility
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                # Should not happen if problem is feasible
                result[i] = 0
    
    return result


def _best_improvement_local_search(result, demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """Best-improvement local search: evaluate all single reassignment moves, take the best.
    Also tries facility closing as a composite move."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = result.copy()
    best_cost = _total_cost(best_result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best_result
    
    load = np.bincount(best_result, weights=demands, minlength=n_fac)
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        best_delta_total = 0.0
        best_move = None  # (type, data)
        
        # --- Phase 1: Evaluate all single reassignment moves ---
        # For each customer i, for each facility f != current, compute delta
        # delta(i, f) = assign_cost[i,f] - assign_cost[i,cur]
        #              + (open_cost[f] if load[f]==0 else 0)
        #              - (open_cost[cur] if load[cur]==demands[i] else 0)
        # Feasible if load[f] + demands[i] <= capacities[f]
        
        # Precompute: for each facility, is it empty?
        is_empty = (load == 0)
        
        for i in range(n_cust):
            cur_f = best_result[i]
            d_i = demands[i]
            
            # Base assignment cost change
            base_delta = assign_cost[i] - assign_cost[i, cur_f]
            
            # Open cost: closing current if this is last customer
            close_cur = open_cost[cur_f] if load[cur_f] == d_i else 0.0
            
            # Open cost: opening target if empty
            open_target = np.where(is_empty, open_cost, 0.0)
            
            # Total delta
            delta = base_delta - close_cur + open_target
            
            # Mask infeasible and same facility
            delta[cur_f] = float('inf')
            infeasible = (load + d_i) > capacities
            delta[infeasible] = float('inf')
            
            if np.any(delta < float('inf')):
                best_f = int(np.argmin(delta))
                d_val = delta[best_f]
                if d_val < best_delta_total - 1e-9:
                    best_delta_total = d_val
                    best_move = ('reassign', i, best_f)
        
        # --- Phase 2: Evaluate facility closing moves ---
        # For each facility f with load > 0, try to move all its customers elsewhere
        # This is expensive, so only do it if we haven't found a good reassignment
        if best_move is None or best_delta_total > -0.1:
            for f in range(n_fac):
                if time.time() - start_time >= time_limit:
                    break
                if load[f] == 0:
                    continue
                custs = np.where(best_result == f)[0]
                if len(custs) == 0:
                    continue
                
                temp_load = load.copy()
                temp_result = best_result.copy()
                feasible = True
                total_delta = 0.0
                
                for i in custs:
                    best_f_new = -1
                    best_d = float('inf')
                    d_i = demands[i]
                    for f2 in range(n_fac):
                        if f2 == f:
                            continue
                        if temp_load[f2] + d_i <= capacities[f2]:
                            dm = assign_cost[i, f2] - assign_cost[i, f]
                            if temp_load[f2] == 0:
                                dm += open_cost[f2]
                            if dm < best_d:
                                best_d = dm
                                best_f_new = f2
                    if best_f_new == -1:
                        feasible = False
                        break
                    temp_load[f] -= d_i
                    temp_load[best_f_new] += d_i
                    temp_result[i] = best_f_new
                    total_delta += best_d
                
                if feasible:
                    total_delta -= open_cost[f]
                    if total_delta < best_delta_total - 1e-9:
                        best_delta_total = total_delta
                        best_move = ('close', f, temp_result, temp_load)
        
        # --- Apply best move ---
        if best_move is not None and best_delta_total < -1e-9:
            if best_move[0] == 'reassign':
                _, i, new_f = best_move
                old_f = best_result[i]
                load[old_f] -= demands[i]
                load[new_f] += demands[i]
                best_result[i] = new_f
            else:  # close
                _, f, new_result, new_load = best_move
                best_result = new_result
                load = new_load
            
            best_cost += best_delta_total
            improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 2:
                break
        else:
            no_improve_count = 0
    
    return best_result


def _first_improvement_local_search(result, demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """First-improvement local search with vectorized reassignment and swaps."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = result.copy()
    best_cost = _total_cost(best_result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best_result
    
    load = np.bincount(best_result, weights=demands, minlength=n_fac)
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Move 1: First-improvement reassignment
        for i in range(n_cust):
            current_f = best_result[i]
            deltas = assign_cost[i] - assign_cost[i, current_f]
            open_change = np.zeros(n_fac)
            if load[current_f] == demands[i]:
                open_change[current_f] -= open_cost[current_f]
            empty_facs = np.where(load == 0)[0]
            open_change[empty_facs] += open_cost[empty_facs]
            deltas += open_change
            mask = np.ones(n_fac, dtype=bool)
            mask[current_f] = False
            mask[load + demands[i] > capacities] = False
            if np.any(mask):
                deltas[~mask] = float('inf')
                best_f = int(np.argmin(deltas))
                if deltas[best_f] < -1e-9:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    best_result[i] = best_f
                    best_cost += deltas[best_f]
                    improved = True
        
        # Move 2: First-improvement swap
        for i in range(n_cust):
            if time.time() - start_time >= time_limit:
                break
            fi = best_result[i]
            valid_j = np.where((best_result != fi) & (np.arange(n_cust) > i))[0]
            if len(valid_j) == 0:
                continue
            fj_arr = best_result[valid_j]
            d_i = demands[i]
            d_j = demands[valid_j]
            feas1 = (load[fi] - d_i + d_j) <= capacities[fi]
            feas2 = (load[fj_arr] - d_j + d_i) <= capacities[fj_arr]
            feas = feas1 & feas2
            if not np.any(feas):
                continue
            valid_j = valid_j[feas]
            fj_arr = best_result[valid_j]
            d_j = demands[valid_j]
            old_cost = assign_cost[i, fi] + assign_cost[valid_j, best_result[valid_j]]
            new_cost = assign_cost[i, fj_arr] + assign_cost[valid_j, fi]
            delta_arr = new_cost - old_cost
            best_idx = int(np.argmin(delta_arr))
            if delta_arr[best_idx] < -1e-9:
                j = valid_j[best_idx]
                fj = fj_arr[best_idx]
                load[fi] += demands[j] - d_i
                load[fj] += d_i - demands[j]
                best_result[i], best_result[j] = fj, fi
                best_cost += delta_arr[best_idx]
                improved = True
        
        # Move 3: Facility closing
        for f in range(n_fac):
            if time.time() - start_time >= time_limit:
                break
            if load[f] == 0:
                continue
            custs_at_f = np.where(best_result == f)[0]
            if len(custs_at_f) == 0:
                continue
            
            temp_load = load.copy()
            temp_result = best_result.copy()
            feasible = True
            total_delta = 0.0
            
            for i in custs_at_f:
                best_f_new = -1
                best_delta = float('inf')
                for f2 in range(n_fac):
                    if f2 == f:
                        continue
                    if temp_load[f2] + demands[i] <= capacities[f2]:
                        delta_move = assign_cost[i, f2] - assign_cost[i, f]
                        if temp_load[f2] == 0:
                            delta_move += open_cost[f2]
                        if delta_move < best_delta:
                            best_delta = delta_move
                            best_f_new = f2
                if best_f_new == -1:
                    feasible = False
                    break
                temp_load[f] -= demands[i]
                temp_load[best_f_new] += demands[i]
                temp_result[i] = best_f_new
                total_delta += best_delta
            
            if feasible:
                total_delta -= open_cost[f]
                if total_delta < -1e-9:
                    load = temp_load
                    best_result = temp_result
                    best_cost += total_delta
                    improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 3:
                break
        else:
            no_improve_count = 0
    
    return best_result


def _pairwise_swap_improve(result, demands, capacities, open_cost, assign_cost, time_limit=5.0):
    """Pairwise swap search with random sampling."""
    n_cust = len(demands)
    n_fac = len(capacities)
    best_result = result.copy()
    best_cost = _total_cost(best_result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best_result
    
    load = np.bincount(best_result, weights=demands, minlength=n_fac)
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        n_samples = 3000
        for _ in range(n_samples):
            if time.time() - start_time >= time_limit:
                break
            i, j = np.random.choice(n_cust, size=2, replace=False)
            if i == j:
                continue
            fi = best_result[i]
            fj = best_result[j]
            if fi == fj:
                continue
            
            d_i = demands[i]
            d_j = demands[j]
            
            if load[fi] - d_i + d_j > capacities[fi]:
                continue
            if load[fj] - d_j + d_i > capacities[fj]:
                continue
            
            old_cost = assign_cost[i, fi] + assign_cost[j, fj]
            new_cost = assign_cost[i, fj] + assign_cost[j, fi]
            delta = new_cost - old_cost
            
            if load[fi] == d_i:
                delta -= open_cost[fi]
            if load[fj] == 0:
                delta += open_cost[fj]
            if load[fj] == d_j:
                delta -= open_cost[fj]
            if load[fi] == 0:
                delta += open_cost[fi]
            
            if delta < -1e-9:
                load[fi] += d_j - d_i
                load[fj] += d_i - d_j
                best_result[i], best_result[j] = fj, fi
                best_cost += delta
                improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                break
        else:
            no_improve_count = 0
    
    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities with multi-start local search."""
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_result = None
    best_cost = float('inf')
    
    # Multi-start with different open penalties
    open_penalties = [0.0, 15.0, 40.0, 80.0, 130.0]
    
    for penalty in open_penalties:
        if time.time() > deadline - 25.0:
            break
        
        result = _greedy_construct(demands, capacities, open_cost, assign_cost,
                                   open_penalty=penalty)
        
        time_left = deadline - time.time() - 15.0
        if time_left < 1.0:
            cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
            continue
        
        # First-improvement local search (fast)
        result = _first_improvement_local_search(result, demands, capacities, open_cost, assign_cost,
                                                 time_limit=time_left * 0.6)
        
        cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Best-improvement polish on the best solution found
    if best_result is not None and time.time() < deadline - 10.0:
        time_left = deadline - time.time() - 5.0
        if time_left > 1.0:
            polished = _best_improvement_local_search(best_result, demands, capacities, open_cost, assign_cost,
                                                      time_limit=time_left)
            cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = polished
    
    # Pairwise swap polish
    if best_result is not None and time.time() < deadline - 4.0:
        time_left = deadline - time.time() - 2.0
        if time_left > 0.5:
            swapped = _pairwise_swap_improve(best_result, demands, capacities, open_cost, assign_cost,
                                             time_limit=time_left)
            cost = _total_cost(swapped, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = swapped
    
    # Final first-improvement pass if time remains
    if best_result is not None and time.time() < deadline - 2.0:
        time_left = deadline - time.time() - 1.0
        if time_left > 0.5:
            final = _first_improvement_local_search(best_result, demands, capacities, open_cost, assign_cost,
                                                    time_limit=time_left)
            cost = _total_cost(final, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = final
    
    if best_result is None:
        best_result = _greedy_construct(demands, capacities, open_cost, assign_cost)
    
    return best_result
# EVOLVE-BLOCK-END
