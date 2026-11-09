import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    # Greedy constructive heuristic
    def greedy_construct():
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        # Process customers in order of largest demand first
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Find feasible facilities
            feasible = []
            for f in range(n_facilities):
                if load[f] + d <= capacities[f]:
                    # Cost if we assign here
                    if load[f] == 0:
                        cost = open_cost[f] + assign_cost[i, f]
                    else:
                        cost = assign_cost[i, f]
                    feasible.append((cost, f))
            
            if feasible:
                feasible.sort(key=lambda x: x[0])
                f_best = feasible[0][1]
            else:
                # Fallback: facility with most remaining capacity
                remaining = capacities - load
                f_best = int(np.argmax(remaining))
            
            res[i] = f_best
            load[f_best] += d
        
        return res, load
    
    # Calculate total cost
    def total_cost(res, load):
        cost = 0.0
        for f in range(n_facilities):
            if load[f] > 0:
                cost += open_cost[f]
        cost += np.sum(assign_cost[np.arange(n_customers), res])
        return cost
    
    # Local search: move and swap operations
    def local_search(res, load, max_iters=500):
        res = res.copy()
        load = load.copy()
        current_cost = total_cost(res, load)
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iters:
            improved = False
            iteration += 1
            
            # Try moves
            best_delta = 0.0
            best_move = None
            
            for i in range(n_customers):
                f_old = res[i]
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    
                    # Compute delta
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
                f_old = res[i]
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                res[i] = f_new
                current_cost += best_delta
                improved = True
                continue
            
            # Try swaps
            best_delta_swap = 0.0
            best_swap = None
            
            for i in range(n_customers):
                f_i = res[i]
                for j in range(i + 1, n_customers):
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                    # Check capacity
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                            (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < best_delta_swap:
                        best_delta_swap = delta
                        best_swap = (i, j)
            
            if best_swap is not None:
                i, j = best_swap
                f_i = res[i]
                f_j = res[j]
                load[f_i] -= demands[i]
                load[f_j] -= demands[j]
                load[f_i] += demands[j]
                load[f_j] += demands[i]
                res[i] = f_j
                res[j] = f_i
                current_cost += best_delta_swap
                improved = True
        
        return res, load, current_cost
    
    # GRASP with multiple restarts
    start_time = time.time()
    time_limit = 110.0  # Leave margin under 120s
    
    best_result = None
    best_cost = float('inf')
    rng = np.random.RandomState(42)
    
    # First, get a good solution with greedy + local search
    res, load = greedy_construct()
    res, load, cost = local_search(res, load)
    if cost < best_cost:
        best_cost = cost
        best_result = res.copy()
    
    # Now do GRASP iterations with randomized construction
    while time.time() - start_time < time_limit:
        # Randomized greedy construction
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            feasible = []
            for f in range(n_facilities):
                if load[f] + d <= capacities[f]:
                    if load[f] == 0:
                        cost_val = open_cost[f] + assign_cost[i, f]
                    else:
                        cost_val = assign_cost[i, f]
                    feasible.append((cost_val, f))
            
            if feasible:
                feasible.sort(key=lambda x: x[0])
                # Random choice from top K
                K = min(3, len(feasible))
                f_best = feasible[rng.randint(K)][1]
            else:
                remaining = capacities - load
                f_best = int(np.argmax(remaining))
            
            res[i] = f_best
            load[f_best] += d
        
        # Local search
        res, load, cost = local_search(res, load)
        
        if cost < best_cost:
            best_cost = cost
            best_result = res.copy()
    
    return best_result
