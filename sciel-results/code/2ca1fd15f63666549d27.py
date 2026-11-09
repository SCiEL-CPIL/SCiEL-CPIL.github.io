"""Capacitated facility location: multi-seed greedy + local search with swaps."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _true_obj(result, demands, capacities, open_cost, assign_cost):
    """Compute objective: open costs + assignment costs."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum


def _greedy_seed(demands, capacities, open_cost, assign_cost, share_param):
    """Largest demand first; choose cheapest facility w.r.t. effective cost."""
    n_fac = len(open_cost)
    n_cust = len(demands)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    # Effective cost: assignment cost + shared open cost per customer
    if share_param > 0:
        eff = assign_cost + share_param * open_cost[None, :] / np.maximum(demands[None, :], 1.0)
    else:
        eff = assign_cost
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        cands = np.argsort(eff[i])
        placed = False
        for f in cands:
            if load[f] + d <= capacities[f]:
                load[f] += d
                result[i] = f
                placed = True
                break
        if not placed:
            for f in np.argsort(assign_cost[i]):
                if load[f] + d <= capacities[f]:
                    load[f] += d
                    result[i] = f
                    placed = True
                    break
    if not all(result >= 0):
        # Fallback: force assignment to any facility with room
        load = np.zeros(n_fac)
        result = np.full(n_cust, -1, dtype=int)
        for i in range(n_cust):
            d = demands[i]
            for f in np.argsort(assign_cost[i]):
                if load[f] + d <= capacities[f]:
                    load[f] += d
                    result[i] = f
                    break
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Local search: single moves and pair swaps until no improvement or deadline."""
    n_cust = len(result)
    n_fac = len(open_cost)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    
    # Compute initial objective
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(n_cust), result]))
    best_obj = open_sum + assign_sum
    
    # Precompute sorted facility orders per customer (by assignment cost)
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    no_improve_rounds = 0
    max_no_improve = 30
    
    while time.time() < deadline and no_improve_rounds < max_no_improve:
        improved = False
        
        # --- Single moves ---
        for c in range(n_cust):
            if time.time() >= deadline:
                break
            d = demands[c]
            cur_f = result[c]
            cur_cost_c = assign_cost[c, cur_f]
            
            for f in sorted_facs[c]:
                if f == cur_f:
                    continue
                if load[f] + d > capacities[f]:
                    continue
                
                delta_assign = assign_cost[c, f] - cur_cost_c
                delta_open = 0.0
                if load[cur_f] == d:  # cur_f becomes empty
                    delta_open -= open_cost[cur_f]
                if load[f] == 0:  # f becomes non-empty
                    delta_open += open_cost[f]
                
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    # Apply move
                    result[c] = f
                    load[cur_f] -= d
                    load[f] += d
                    best_obj += delta
                    improved = True
                    break
        
        # --- Pair swaps ---
        if time.time() < deadline:
            for c1 in range(n_cust):
                if time.time() >= deadline:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                for c2 in range(c1 + 1, n_cust):
                    if time.time() >= deadline:
                        break
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
                        break
                if improved:
                    break
        
        if not improved:
            no_improve_rounds += 1
        else:
            no_improve_rounds = 0
    
    return result, best_obj


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n_cust = len(demands)
    n_fac = len(open_cost)

    t0 = time.time()
    hard_deadline = t0 + 120.0
    safety = 15.0
    deadline = hard_deadline - safety

    best_result = None
    best_val = np.inf

    # Try multiple share parameters for seed diversity
    share_params = [0.0, 0.3, 1.0, 2.0, 5.0]
    for sp in share_params:
        if time.time() > deadline:
            break
        seed = _greedy_seed(demands, capacities, open_cost, assign_cost, sp)
        if np.any(seed < 0):
            continue
        # Verify feasibility
        load_check = np.zeros(n_fac)
        for i in range(n_cust):
            load_check[seed[i]] += demands[i]
        if np.any(load_check > capacities + 1e-9):
            continue
        
        # Local search with remaining time budget
        # Give each seed a portion of the time
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        # Limit local search time per seed
        ls_deadline = min(deadline, time.time() + remaining * 0.8)
        res, val = _local_search(seed, demands, capacities, open_cost, assign_cost, ls_deadline)
        
        if val < best_val:
            best_val = val
            best_result = res.copy()
    
    # If no valid result, use pure greedy
    if best_result is None or np.any(best_result < 0):
        best_result = _greedy_seed(demands, capacities, open_cost, assign_cost, 0.0)

    # Final feasibility enforcement
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = int(best_result[i])
        if f < 0 or f >= n_fac:
            for ff in np.argsort(assign_cost[i]):
                if load[ff] + demands[i] <= capacities[ff]:
                    best_result[i] = ff
                    load[ff] += demands[i]
                    break
        else:
            load[f] += demands[i]

    return np.asarray(best_result, dtype=int)
# EVOLVE-BLOCK-END
