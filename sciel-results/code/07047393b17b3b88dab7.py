import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    def _is_valid(result):
        if len(result) != n_cust:
            return False
        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return False
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return False
        return True

    def _compute_cost(result):
        if not _is_valid(result):
            return float('inf')
        open_facilities = np.unique(result)
        cost = np.sum(open_cost[open_facilities]) + np.sum(assign_cost[np.arange(n_cust), result])
        return cost

    def _greedy_assign(order=None, alpha=0.0):
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        if order is None:
            order = np.argsort(-demands)
            
        for i in order:
            if alpha > 0:
                noise = np.random.uniform(0, alpha * np.max(assign_cost[i]))
                eff_cost = assign_cost[i] + noise
            else:
                eff_cost = assign_cost[i]
                
            fac_order = np.argsort(eff_cost)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
                    
            if not assigned:
                feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += demands[i]
                    result[i] = f_best
                else:
                    result[i] = 0
                    
        return result

    def _local_search(result, time_limit):
        n_cust_local = n_cust
        n_fac_local = n_fac
        current = result.astype(int).copy()
        
        load = np.zeros(n_fac_local)
        fac_count = np.zeros(n_fac_local, dtype=int)
        
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        current_cost = _compute_cost(current)
        if np.isinf(current_cost):
            return result
            
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        no_improve_count = 0
        max_no_improve = 5
        
        while (time.time() - start < time_limit) and no_improve_count < max_no_improve:
            improved_this_pass = False
            cust_order = np.random.permutation(n_cust_local)
            
            # 1-move: single reassignment
            for i in cust_order:
                if time.time() - start > time_limit:
                    return best_result
                
                cur_f = current[i]
                d_i = demands[i]
                
                delta_assign = assign_cost[i, :] - assign_cost[i, cur_f]
                
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[cur_f] = False
                
                if not np.any(feasible_mask):
                    continue
                    
                open_delta = np.zeros(n_fac_local)
                if fac_count[cur_f] == 1:
                    open_delta[:] -= open_cost[cur_f]
                empty_mask = (fac_count == 0)
                open_delta[empty_mask] += open_cost[empty_mask]
                
                total_delta = delta_assign + open_delta
                total_delta[~feasible_mask] = float('inf')
                
                min_idx = np.argmin(total_delta)
                min_val = total_delta[min_idx]
                
                if min_val < -1e-9:
                    new_f = min_idx
                    load[cur_f] -= d_i
                    load[new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[new_f] += 1
                    current[i] = new_f
                    current_cost += min_val
                    improved_this_pass = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                        
            # 2-move: swaps between different facilities
            if time.time() - start < time_limit:
                # Group customers by facility for efficient swap search
                fac_to_custs = {f: [] for f in range(n_fac_local)}
                for i in range(n_cust_local):
                    fac_to_custs[current[i]].append(i)
                
                # Try swaps between customers in different facilities
                for f1 in range(n_fac_local):
                    if len(fac_to_custs[f1]) == 0:
                        continue
                    for f2 in range(f1 + 1, n_fac_local):
                        if len(fac_to_custs[f2]) == 0:
                            continue
                        if time.time() - start > time_limit:
                            return best_result
                        
                        # Sample up to 10 pairs from each facility to limit computation
                        sample1 = fac_to_custs[f1][:10] if len(fac_to_custs[f1]) > 10 else fac_to_custs[f1]
                        sample2 = fac_to_custs[f2][:10] if len(fac_to_custs[f2]) > 10 else fac_to_custs[f2]
                        
                        for i in sample1:
                            for j in sample2:
                                d_i = demands[i]
                                d_j = demands[j]
                                
                                if load[f1] - d_i + d_j > capacities[f1]:
                                    continue
                                if load[f2] - d_j + d_i > capacities[f2]:
                                    continue
                                
                                delta = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                                
                                if delta < -1e-9:
                                    load[f1] = load[f1] - d_i + d_j
                                    load[f2] = load[f2] - d_j + d_i
                                    current[i] = f2
                                    current[j] = f1
                                    current_cost += delta
                                    improved_this_pass = True
                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_result = current.copy()
                                    break
                            if time.time() - start > time_limit:
                                return best_result
                        if time.time() - start > time_limit:
                            return best_result
            
            if not improved_this_pass:
                no_improve_count += 1
            else:
                no_improve_count = 0
                
        return best_result

    # --- Main Logic ---
    
    best_initial = None
    best_initial_cost = float('inf')
    
    # Multiple diverse initial solutions
    orders = [
        np.argsort(-demands),           # Largest first
        np.argsort(demands),            # Smallest first
        np.random.permutation(n_cust),  # Random
        np.argsort(-demands / (assign_cost.min(axis=1) + 1)),  # Demand-to-cost ratio
    ]
    
    for order in orders:
        sol = _greedy_assign(order=order, alpha=0)
        cost = _compute_cost(sol)
        if cost < best_initial_cost:
            best_initial = sol
            best_initial_cost = cost
    
    # Stochastic greedy for diversity
    for _ in range(8):
        sol_stoch = _greedy_assign(order=np.argsort(-demands), alpha=np.random.uniform(0.05, 0.2))
        cost_stoch = _compute_cost(sol_stoch)
        if cost_stoch < best_initial_cost:
            best_initial = sol_stoch
            best_initial_cost = cost_stoch

    remaining_time = deadline - time.time()
    if remaining_time > 2 and best_initial is not None:
        final_result = _local_search(best_initial, remaining_time - 1)
    else:
        final_result = best_initial
        
    if final_result is None:
        return np.zeros(n_cust, dtype=int)
        
    if not _is_valid(final_result):
        if best_initial is not None and _is_valid(best_initial):
            return best_initial
        else:
            return np.zeros(n_cust, dtype=int)
            
    return final_result
