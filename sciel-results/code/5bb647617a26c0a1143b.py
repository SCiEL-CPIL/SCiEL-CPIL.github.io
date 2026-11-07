import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Phase 1: Initial greedy assignment - largest demand first
    order = np.argsort(-demands)
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    open_facilities = np.zeros(n_facilities, dtype=bool)
    
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
            result[i] = f
            load[f] += demands[i]
            open_facilities[f] = True
            continue
        
        best_f = -1
        best_score = np.inf
        for f in feasible:
            if open_facilities[f]:
                score = assign_cost[i, f]
            else:
                score = open_cost[f] + assign_cost[i, f]
            if score < best_score:
                best_score = score
                best_f = f
        
        result[i] = best_f
        load[best_f] += demands[i]
        open_facilities[best_f] = True
    
    # Phase 2: Local search
    current_cost = 0.0
    open_set = set(result.tolist())
    for f in open_set:
        current_cost += open_cost[f]
    for i in range(n_customers):
        current_cost += assign_cost[i, result[i]]
    
    improved = True
    max_rounds = 300
    for _ in range(max_rounds):
        if not improved:
            break
        improved = False
        
        # Single moves
        for i in range(n_customers):
            cur_f = result[i]
            for new_f in range(n_facilities):
                if new_f == cur_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                load[cur_f] -= demands[i]
                cur_becomes_empty = (load[cur_f] == 0)
                new_was_empty = (load[new_f] == 0)
                load[new_f] += demands[i]
                
                cost_delta = 0.0
                if new_was_empty:
                    cost_delta += open_cost[new_f]
                if cur_becomes_empty:
                    cost_delta -= open_cost[cur_f]
                cost_delta += assign_cost[i, new_f] - assign_cost[i, cur_f]
                
                if cost_delta < -1e-9:
                    result[i] = new_f
                    current_cost += cost_delta
                    if new_was_empty:
                        open_set.add(new_f)
                    if cur_becomes_empty:
                        open_set.discard(cur_f)
                    improved = True
                else:
                    load[cur_f] += demands[i]
                    load[new_f] -= demands[i]
        
        # Pair swaps
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                
                new_load_fi = load[fi] - demands[i] + demands[j]
                new_load_fj = load[fj] - demands[j] + demands[i]
                
                if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                    continue
                
                fi_empty_after = (new_load_fi == 0)
                fj_empty_after = (new_load_fj == 0)
                
                cost_delta = 0.0
                if fj_empty_after:
                    cost_delta -= open_cost[fj]
                if fi_empty_after:
                    cost_delta -= open_cost[fi]
                cost_delta += assign_cost[i, fj] - assign_cost[i, fi]
                cost_delta += assign_cost[j, fi] - assign_cost[j, fj]
                
                if cost_delta < -1e-9:
                    result[i], result[j] = fj, fi
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    current_cost += cost_delta
                    if fj_empty_after:
                        open_set.discard(fj)
                    if fi_empty_after:
                        open_set.discard(fi)
                    improved = True
    
    return result
