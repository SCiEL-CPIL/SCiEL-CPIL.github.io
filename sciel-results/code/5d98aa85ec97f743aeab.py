# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_init(demands, capacities, assign_cost, allowed_facilities):
    """
    Greedy assignment to allowed facilities.
    """
    n_cust = len(demands)
    n_allowed = len(allowed_facilities)
    if n_allowed == 0:
        return None
    
    # Map local index to global facility index
    # We work with global indices for load tracking to be consistent
    load = np.zeros(len(capacities))
    assignment = np.full(n_cust, -1, dtype=int)
    
    # Order customers by demand descending
    order = np.argsort(-demands)
    
    # Precompute costs for allowed facilities
    # allowed_facilities is a list/array of global indices
    costs_allowed = assign_cost[:, allowed_facilities]
    # Sort facilities for each customer by cost
    # This is (n_cust, n_allowed)
    sorted_local_facs = np.argsort(costs_allowed, axis=1)
    
    for i in order:
        placed = False
        for f_local in sorted_local_facs[i]:
            f_global = allowed_facilities[f_local]
            if load[f_global] + demands[i] <= capacities[f_global]:
                load[f_global] += demands[i]
                assignment[i] = f_global
                placed = True
                break
        if not placed:
            return None
            
    return assignment

def _global_local_search(assignment, demands, capacities, open_cost, assign_cost, max_iters=500):
    """
    Local search allowing opening/closing facilities.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = assignment[i]
        load[f] += demands[i]
        
    current_cost = sum(open_cost[f] for f in range(n_fac) if load[f] > 0) + \
                   sum(assign_cost[i, assignment[i]] for i in range(n_cust))
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # 1. Single Moves
        for i in range(n_cust):
            f_old = assignment[i]
            d_i = demands[i]
            cost_old = assign_cost[i, f_old]
            
            for f_new in range(n_fac):
                if f_new == f_old:
                    continue
                
                if load[f_new] + d_i > capacities[f_new]:
                    continue
                
                delta_assign = assign_cost[i, f_new] - cost_old
                
                delta_open = 0
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                if load[f_old] == d_i:
                    delta_open -= open_cost[f_old]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    load[f_old] -= d_i
                    load[f_new] += d_i
                    assignment[i] = f_new
                    current_cost += total_delta
                    improved = True
                    break
        
        # 2. Swaps
        if not improved:
            for i in range(n_cust):
                f_i = assignment[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_cust):
                    f_j = assignment[j]
                    d_j = demands[j]
                    
                    if f_i == f_j:
                        continue
                    
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost doesn't change for swaps between occupied facilities
                    if delta_assign < -1e-9:
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        assignment[i] = f_j
                        assignment[j] = f_i
                        current_cost += delta_assign
                        improved = True
                        break
                if improved:
                    break
                    
    return assignment, current_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_assignment = None
    best_cost = float('inf')
    
    # Generate candidate subsets
    subsets = []
    
    # 1. All facilities
    subsets.append(np.arange(n_fac))
    
    # 2. Top K by open cost
    by_open = np.argsort(open_cost)
    for k in [10, 15, 20]:
        if k <= n_fac:
            subsets.append(by_open[:k])
            
    # 3. Top K by avg assign cost
    avg_assign = assign_cost.mean(axis=0)
    by_assign = np.argsort(avg_assign)
    for k in [10, 15, 20]:
        if k <= n_fac:
            subsets.append(by_assign[:k])
            
    # 4. Random subsets
    np.random.seed(42)
    for _ in range(5):
        k = np.random.randint(10, n_fac + 1)
        subsets.append(np.random.choice(n_fac, k, replace=False))
        
    # Evaluate candidates
    for subset in subsets:
        if time.time() > deadline - 20:
            break
            
        init_assign = _greedy_init(demands, capacities, assign_cost, subset)
        if init_assign is None:
            continue
            
        # Run local search
        refined_assign, cost = _global_local_search(init_assign, demands, capacities, open_cost, assign_cost, max_iters=100)
        
        if cost < best_cost:
            best_cost = cost
            best_assignment = refined_assign.copy()
            
    # Final polish on the best one
    if best_assignment is not None:
        if time.time() < deadline - 5:
            refined_assign, cost = _global_local_search(best_assignment, demands, capacities, open_cost, assign_cost, max_iters=500)
            if cost < best_cost:
                best_cost = cost
                best_assignment = refined_assign.copy()
                
    # Fallback
    if best_assignment is None:
        best_assignment = _greedy_init(demands, capacities, assign_cost, np.arange(n_fac))
        if best_assignment is None:
            # Should not happen if total capacity is sufficient
            best_assignment = np.zeros(n_cust, dtype=int)
            
    return best_assignment
# EVOLVE-BLOCK-END
