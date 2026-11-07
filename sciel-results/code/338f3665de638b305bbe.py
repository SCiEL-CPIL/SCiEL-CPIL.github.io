import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    best_result = None
    best_score = float('inf')
    
    def evaluate(result):
        # Validate indices
        if np.any(result < 0) or np.any(result >= n_facilities):
            return float('inf')
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                # Fallback: pick cheapest feasible if any, else cheapest overall
                feasible_any = np.where(capacities >= 0)[0]
                if len(feasible_any) == 0:
                    f = np.argmin(assign_cost[i])
                else:
                    # Try to find any facility with remaining capacity
                    rem_cap = capacities - load
                    pos = rem_cap > 0
                    if np.any(pos):
                        candidates = np.where(pos)[0]
                    else:
                        candidates = feasible_any
                    f = candidates[np.argmin(assign_cost[i, candidates])]
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                f = top_k[np.random.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open

    def local_search(result, load, is_open):
        improved = True
        while improved:
            improved = False
            # Iterate customers in random order to break symmetry
            order = np.random.permutation(n_customers)
            for i in order:
                f_old = result[i]
                d = demands[i]
                
                best_delta = 1e-9
                best_f_new = -1
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    if load[f_new] + d > capacities[f_new] + 1e-9:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    if load[f_old] - d == 0:
                        delta_open -= open_cost[f_old]
                    
                    if load[f_new] == 0:
                        if not is_open[f_new]:
                            delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                    load[best_f_new] += d
                    if load[best_f_new] == d:
                        is_open[best_f_new] = True
                    result[i] = best_f_new
                    improved = True
                    
                    if time.time() > deadline:
                        return result, load, is_open
                        
        return result, load, is_open

    # 1. Deterministic greedy
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()
    
    # 2. Randomized restarts
    for k in range(80):
        if time.time() > deadline - 5:
            break
        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = local_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
    
    return best_result
