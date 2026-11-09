"""Capacitated facility location: vectorized heuristic + local search with time budget."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _true_obj(result, demands, capacities, open_cost, assign_cost):
    """Compute objective: open costs + assignment costs. Returns inf if infeasible."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum


def _greedy_seed(demands, capacities, assign_cost):
    """Largest-first greedy assignment (fallback)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                result[i] = f
                load[f] += d
                break
        if result[i] == -1:
            result[i] = 0
            load[0] += d
    return result


def _vectorized_seed(demands, capacities, assign_cost):
    """Vectorized heuristic: assign all customers to their cheapest facility, then repair capacity violations."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    result = np.argmin(assign_cost, axis=1)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    
    for _ in range(n_fac * 5):
        excess_mask = load > capacities
        if not np.any(excess_mask):
            break
        
        for f in np.where(excess_mask)[0]:
            while load[f] > capacities[f]:
                customers_at_f = np.where(result == f)[0]
                if len(customers_at_f) == 0:
                    break
                
                c = customers_at_f[np.argmax(demands[customers_at_f])]
                d = demands[c]
                
                alt_costs = assign_cost[c].copy()
                alt_costs[f] = np.inf
                alt_order = np.argsort(alt_costs)
                
                moved = False
                for af in alt_order:
                    if load[af] + d <= capacities[af]:
                        load[f] -= d
                        load[af] += d
                        result[c] = af
                        moved = True
                        break
                
                if not moved:
                    break
    
    if np.any(load > capacities):
        return _greedy_seed(demands, capacities, assign_cost)
    
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Local search: vectorized single moves + pair swaps."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    
    best_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if best_obj == float('inf'):
        return result, best_obj
    
    no_improve = 0
    max_no_improve = 150
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # --- Single Moves (Vectorized Selection) ---
        cur_costs = assign_cost[np.arange(n_cust), result]
        
        diff = assign_cost - cur_costs[:, np.newaxis]
        
        for c in range(n_cust):
            diff[c, result[c]] = np.inf
            
        new_loads = load[np.newaxis, :] + demands[:, np.newaxis]
        feasible = new_loads <= capacities[np.newaxis, :]
        
        diff[~feasible] = np.inf
        
        src_empty = (load[result] == demands)
        tgt_empty = (load == 0)
        
        open_cost_src = np.where(src_empty, -open_cost[result], 0.0)
        open_cost_tgt = np.where(tgt_empty, open_cost, 0.0)
        
        delta_matrix = diff + open_cost_src[:, np.newaxis] + open_cost_tgt[np.newaxis, :]
        
        best_f = np.argmin(delta_matrix, axis=1)
        best_delta = np.min(delta_matrix, axis=1)
        
        candidates = np.where(best_delta < -1e-9)[0]
        if len(candidates) > 0:
            candidates = candidates[np.argsort(best_delta[candidates])]
            
            for c in candidates:
                if time.time() >= deadline:
                    break
                f_new = best_f[c]
                f_old = result[c]
                d = demands[c]
                
                if load[f_new] + d <= capacities[f_new]:
                    delta = best_delta[c]
                    load[f_old] -= d
                    load[f_new] += d
                    result[c] = f_new
                    best_obj += delta
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            
            swap_improved = False
            for c1 in range(n_cust):
                if time.time() >= deadline:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                for c2 in range(c1 + 1, n_cust):
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]
                    
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    
                    if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                        continue
                    
                    delta_assign = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                   (assign_cost[c1, f1] + assign_cost[c2, f2])
                    
                    delta_open = 0.0
                    if load[f1] > 0 and new_load_f1 == 0:
                        delta_open -= open_cost[f1]
                    elif load[f1] == 0 and new_load_f1 > 0:
                        delta_open += open_cost[f1]
                        
                    if load[f2] > 0 and new_load_f2 == 0:
                        delta_open -= open_cost[f2]
                    elif load[f2] == 0 and new_load_f2 > 0:
                        delta_open += open_cost[f2]
                    
                    delta = delta_assign + delta_open
                    
                    if delta < -1e-9:
                        result[c1], result[c2] = f2, f1
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        best_obj += delta
                        improved = True
                        swap_improved = True
                        no_improve = 0
                        break
                if swap_improved:
                    break
            
            if not swap_improved and not improved:
                no_improve += 1
                if no_improve > 15:
                    break
        else:
            no_improve = 0
    
    return result, best_obj


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using vectorized seed + local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    deadline = time.time() + 145.0
    
    result = _vectorized_seed(demands, capacities, assign_cost)
    
    result, best_obj = _local_search(result, demands, capacities, open_cost, assign_cost, deadline)
    
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return _greedy_seed(demands, capacities, assign_cost)
    
    return result.astype(int)
# EVOLVE-BLOCK-END
