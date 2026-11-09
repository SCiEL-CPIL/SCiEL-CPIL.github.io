import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 120.0  # 20s margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = None
    best_cost = float('inf')
    
    def calc_cost(res):
        total = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for i in range(n_customers):
            f = res[i]
            total += assign_cost[i, f]
            used[f] = True
        total += np.sum(open_cost[used])
        return total

    def local_search(result, load):
        # result: array of facility indices
        # load: array of current loads
        current_cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for i in range(n_customers):
            f = result[i]
            current_cost += assign_cost[i, f]
            used[f] = True
        current_cost += np.sum(open_cost[used])
        
        # Pre-compute sorted facilities for each customer for potential speedup?
        # But we check all facilities in the loop below.
        
        while time.time() < deadline:
            best_delta = 0
            best_move = None
            
            # Shuffle customers to avoid bias? Or keep fixed order.
            # Fixed order is deterministic. Shuffling might help escape symmetry.
            # But for "Best Improvement", order doesn't matter for the final step,
            # only for which move is picked if there are ties.
            
            for i in range(n_customers):
                f_old = result[i]
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    
                    delta = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    if load[f_old] == demands[i]:
                        delta -= open_cost[f_old]
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, f_new)
            
            if best_move is not None:
                i, f_new = best_move
                f_old = result[i]
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                current_cost += best_delta
            else:
                break
                
        return result, current_cost

    # Strategy 1: Greedy Init
    order = np.argsort(-demands)
    result1 = np.zeros(n_customers, dtype=int)
    load1 = np.zeros(n_facilities)
    sorted_facs = np.argsort(assign_cost, axis=1)
    for i in order:
        for f in sorted_facs[i]:
            if load1[f] + demands[i] <= capacities[f]:
                load1[f] += demands[i]
                result1[i] = f
                break
        
    res1, cost1 = local_search(result1.copy(), load1.copy())
    if cost1 < best_cost:
        best_cost = cost1
        best_result = res1.copy()

    # Strategy 2: Random Restarts (if time permits)
    # We can try a few random feasible starts.
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10: # Stop if less than 10s left
            break
            
        # Generate random feasible assignment
        res_r = np.zeros(n_customers, dtype=int)
        load_r = np.zeros(n_facilities)
        feasible = True
        # Process larger demands first to increase feasibility chance
        random_order = np.argsort(-demands)
        # Add some randomness to the order?
        # Or just random assignment to feasible facilities.
        
        for i in random_order:
            # Find all feasible facilities
            feasible_facs = []
            for f in range(n_facilities):
                if load_r[f] + demands[i] <= capacities[f]:
                    feasible_facs.append(f)
            if not feasible_facs:
                feasible = False
                break
            # Pick one randomly
            f = feasible_facs[random.randint(0, len(feasible_facs)-1)]
            load_r[f] += demands[i]
            res_r[i] = f
            
        if not feasible:
            continue
            
        res_r, cost_r = local_search(res_r, load_r)
        if cost_r < best_cost:
            best_cost = cost_r
            best_result = res_r.copy()

    if best_result is None:
        # Fallback to simple greedy if something went wrong
        return result1
        
    return best_result
