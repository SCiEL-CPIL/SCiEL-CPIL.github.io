"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 145.0  # Safety margin below 150s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    def time_exceeded():
        return time.time() - start_time > timeout
    
    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum
    
    def greedy_init(lam):
        """Greedy init with Lagrangian-adjusted costs"""
        order = np.argsort(-demands)
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        # Adjusted cost includes Lagrangian multiplier for capacity
        # Effective cost: assign_cost[i,f] + lam * (1/capacity[f] if capacity > 0 else 0)
        # Actually, let's use a simpler approach: just adjust by lam * load/capacity
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                f = np.argmax(remaining_cap)
            else:
                # Use adjusted cost
                adj_costs = assign_cost[i].copy()
                for f in range(n_facilities):
                    if feasible_mask[f]:
                        # Penalize facilities that are already heavily loaded
                        adj_costs[f] += lam * (load[f] / max(capacities[f], 1))
                    else:
                        adj_costs[f] = np.inf
                f = np.argmin(adj_costs)
            
            res[i] = f
            load[f] += d
        
        return res, load
    
    def local_search(result, load, max_rounds=100):
        """Local search with 1-move and 2-move (swap) improvements"""
        current_cost = calc_total_cost(result, load)
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()
        
        round_num = 0
        while round_num < max_rounds:
            if time_exceeded():
                break
            
            round_num += 1
            improved = False
            best_delta = 0.0
            best_move_type = None
            best_move_data = None
            
            # Phase A: 1-moves (Relocation)
            for i in range(n_customers):
                if time_exceeded():
                    break
                
                f_old = result[i]
                d = demands[i]
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    if load[f_new] + d > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    
                    if load[f_old] == d:
                        delta_open -= open_cost[f_old]
                    
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move_type = 1
                        best_move_data = (i, f_old, f_new)
            
            # Phase B: 2-moves (Swap)
            for i in range(n_customers):
                if time_exceeded():
                    break
                
                f_i = result[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_customers):
                    if time_exceeded():
                        break
                    
                    f_j = result[j]
                    d_j = demands[j]
                    
                    if f_i == f_j:
                        continue
                    
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    old_open = (open_cost[f_i] if load[f_i] > 0 else 0.0) + \
                               (open_cost[f_j] if load[f_j] > 0 else 0.0)
                    
                    new_open = (open_cost[f_i] if new_load_i > 0 else 0.0) + \
                               (open_cost[f_j] if new_load_j > 0 else 0.0)
                    
                    delta_open = new_open - old_open
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move_type = 2
                        best_move_data = (i, j, f_i, f_j)
            
            # Apply best move if found
            if best_move_type is not None:
                if best_move_type == 1:
                    i, f_old, f_new = best_move_data
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    result[i] = f_new
                else:
                    i, j, f_i, f_j = best_move_data
                    d_i = demands[i]
                    d_j = demands[j]
                    load[f_i] -= d_i
                    load[f_j] -= d_j
                    load[f_i] += d_j
                    load[f_j] += d_i
                    result[i] = f_j
                    result[j] = f_i
                
                current_cost += best_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
                improved = True
            else:
                break
        
        return best_result, best_load, best_cost
    
    def sa_local_search(result, load, temp_init=1.0, cooling=0.995, max_iter=5000):
        """Simulated Annealing local search"""
        current_cost = calc_total_cost(result, load)
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()
        
        temp = temp_init
        iter_num = 0
        while iter_num < max_iter:
            if time_exceeded():
                break
            
            iter_num += 1
            temp *= cooling
            
            # Random move
            move_type = np.random.random()
            
            if move_type < 0.5:
                # Relocation
                i = np.random.randint(n_customers)
                f_old = result[i]
                d = demands[i]
                
                feasible = np.where((capacities - load) >= d)[0]
                feasible = feasible[feasible != f_old]
                
                if len(feasible) == 0:
                    continue
                
                f_new = feasible[np.random.randint(len(feasible))]
                
                old_cost = current_cost
                load[f_old] -= d
                load[f_new] += d
                result[i] = f_new
                new_cost = calc_total_cost(result, load)
                
                delta = new_cost - old_cost
                if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = result.copy()
                        best_load = load.copy()
                else:
                    # Reject
                    load[f_old] += d
                    load[f_new] -= d
                    result[i] = f_old
            
            else:
                # Swap
                i = np.random.randint(n_customers)
                j = np.random.randint(n_customers)
                
                if i == j or result[i] == result[j]:
                    continue
                
                f_i = result[i]
                f_j = result[j]
                d_i = demands[i]
                d_j = demands[j]
                
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                
                if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                    continue
                
                old_cost = current_cost
                load[f_i] -= d_i
                load[f_j] -= d_j
                load[f_i] += d_j
                load[f_j] += d_i
                result[i] = f_j
                result[j] = f_i
                new_cost = calc_total_cost(result, load)
                
                delta = new_cost - old_cost
                if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = result.copy()
                        best_load = load.copy()
                else:
                    # Reject
                    load[f_i] -= d_j
                    load[f_j] -= d_i
                    load[f_i] += d_i
                    load[f_j] += d_j
                    result[i] = f_i
                    result[j] = f_j
        
        return best_result, best_load, best_cost
    
    # Multiple strategies with different Lagrangian multipliers
    best_result = None
    best_load = None
    best_cost = np.inf
    
    # Try several Lagrangian multiplier values
    lam_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    for lam in lam_values:
        if time_exceeded():
            break
        
        # Greedy init with this lambda
        result, load = greedy_init(lam)
        
        # Run local search
        res, load, cost = local_search(result, load)
        
        if cost < best_cost:
            best_result = res
            best_load = load
            best_cost = cost
    
    # If time permits, run SA on the best solution found
    if not time_exceeded() and best_result is not None:
        res, load, cost = sa_local_search(best_result.copy(), best_load.copy(), 
                                           temp_init=2.0, cooling=0.999, max_iter=10000)
        if cost < best_cost:
            best_result = res
            best_load = load
            best_cost = cost
    
    # Random restarts if time permits
    rng = np.random.default_rng(42)
    restart_count = 0
    max_restarts = 20
    
    while not time_exceeded() and restart_count < max_restarts:
        # Random perturbation of current best solution
        result = best_result.copy()
        load = best_load.copy()
        
        # Perform 5-15 random swaps/relocations
        num_perturbations = rng.integers(5, 16)
        for _ in range(num_perturbations):
            if time_exceeded():
                break
            
            move_type = rng.integers(0, 2)  # 0 = relocation, 1 = swap
            
            if move_type == 0:
                # Relocation
                i = rng.integers(0, n_customers)
                f_old = result[i]
                d = demands[i]
                
                feasible = np.where((capacities - load) >= d)[0]
                feasible = feasible[feasible != f_old]
                
                if len(feasible) > 0:
                    f_new = feasible[rng.integers(0, len(feasible))]
                    load[f_old] -= d
                    load[f_new] += d
                    result[i] = f_new
            
            else:
                # Swap
                i = rng.integers(0, n_customers)
                j = rng.integers(0, n_customers)
                
                if i == j or result[i] == result[j]:
                    continue
                
                f_i = result[i]
                f_j = result[j]
                d_i = demands[i]
                d_j = demands[j]
                
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                
                if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                    continue
                
                load[f_i] -= d_i
                load[f_j] -= d_j
                load[f_i] += d_j
                load[f_j] += d_i
                result[i] = f_j
                result[j] = f_i
        
        # Run local search from perturbed solution
        res, load, cost = local_search(result, load)
        
        if cost < best_cost:
            best_result = res
            best_load = load
            best_cost = cost
        
        restart_count += 1
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
