"""Capacitated facility location: Multi-start with swap moves and 2-opt."""
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
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Initialize a valid solution using a greedy heuristic (fast)
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            remaining = capacities - load
            f = np.argmax(remaining)
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]
            
    best_result = result_greedy.copy()
    used_fac = np.unique(best_result)
    score_greedy = np.sum(open_cost[used_fac]) + np.sum(assign_cost[np.arange(n_cust), best_result])
    best_score = score_greedy

    def calc_cost(res):
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def local_search(res, load):
        """Local search with single moves, swap moves, and 2-opt moves."""
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            
            # Single customer moves
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                    
                f_old = res[i]
                best_delta = np.inf
                best_f_new = -1
                
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    if load[f_old] == demands[i]:
                        delta_open -= open_cost[f_old]
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                
                if best_f_new != -1 and best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= demands[i]
                    load[best_f_new] += demands[i]
                    improved = True
            
            # 2-opt moves: move customer to a facility that already has at least one customer
            # This avoids the open cost penalty and can find better assignments
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                    
                f_old = res[i]
                best_delta = np.inf
                best_f_new = -1
                
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    if load[f_new] == 0:
                        continue  # Skip if it would open a new facility
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    if load[f_old] == demands[i]:
                        delta_open -= open_cost[f_old]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                
                if best_f_new != -1 and best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= demands[i]
                    load[best_f_new] += demands[i]
                    improved = True
            
            # Swap moves: try swapping two customers
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() > deadline - 2.0:
                        break
                    
                    f_i = res[i]
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                    
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                        continue
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    delta_open = 0.0
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        res[i], res[j] = res[j], res[i]
                        load[f_i] -= demands[i]
                        load[f_i] += demands[j]
                        load[f_j] -= demands[j]
                        load[f_j] += demands[i]
                        improved = True
                        
        return res, load
    
    # Multi-start with different alpha penalties
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
            
        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1)
        
        res = init_res.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
            
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
            
        if np.all(load <= capacities):
            used = np.unique(res)
            score = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
            if score < best_score:
                best_score = score
                best_result = res.copy()
                
                # Run local search on the best result
                res_ls, load_ls = local_search(best_result.copy(), load.copy())
                if np.all(load_ls <= capacities):
                    used_ls = np.unique(res_ls)
                    score_ls = np.sum(open_cost[used_ls]) + np.sum(assign_cost[np.arange(n_cust), res_ls])
                    if score_ls < best_score:
                        best_score = score_ls
                        best_result = res_ls.copy()

    # Final local search on the best result found
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[best_result[i]] += demands[i]
    
    res_final, load_final = local_search(best_result.copy(), current_load.copy())
    if np.all(load_final <= capacities):
        used_final = np.unique(res_final)
        score_final = np.sum(open_cost[used_final]) + np.sum(assign_cost[np.arange(n_cust), res_final])
        if score_final < best_score:
            best_result = res_final
            best_score = score_final

    return best_result
# EVOLVE-BLOCK-END
