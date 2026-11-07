"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute the total cost of a given assignment."""
    if result is None:
        return np.inf
    
    n_customers = len(result)
    n_facilities = len(capacities)
    
    # Check validity
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return np.inf
        load[f] += demands[i]
        
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return np.inf
            
    total_cost = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            total_cost += open_cost[f]
            
    total_cost += np.sum(assign_cost[np.arange(n_customers), result])
    return total_cost

def _greedy_init(demands, capacities, open_cost, assign_cost):
    """
    Greedy initialization: 
    1. Sort customers by demand descending.
    2. For each customer, pick the facility that minimizes (assign_cost + marginal_open_cost).
    Marginal open cost is open_cost[f] / capacity[f] if facility is empty, else 0.
    This is a heuristic to balance load and cost.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    open_status = np.zeros(n_facilities, dtype=bool)
    
    # Order customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                # Calculate incremental cost
                # If facility is not open, we pay open_cost[f]
                # If it is open, we pay 0
                # We also pay assign_cost[i, f]
                inc_open = open_cost[f] if not open_status[f] else 0.0
                total_inc = assign_cost[i, f] + inc_open
                if total_inc < best_cost:
                    best_cost = total_inc
                    best_f = f
        
        if best_f != -1:
            result[i] = best_f
            load[best_f] += demands[i]
            open_status[best_f] = True
        else:
            # Fallback: should not happen if total capacity > total demand
            # Assign to facility with most remaining capacity
            rem_cap = capacities - load
            best_f = np.argmax(rem_cap)
            result[i] = best_f
            load[best_f] += demands[i]
            open_status[best_f] = True
            
    return result

def _lagrangian_init(demands, capacities, open_cost, assign_cost, steps=100):
    """
    Initialize using Lagrangian relaxation.
    Relax capacity constraints with dual variables lambda.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initialize lambda to 0
    lambda_vec = np.zeros(n_facilities)
    step_size = 1.0
    best_result = None
    best_cost = np.inf
    
    for _ in range(steps):
        # 1. Solve relaxed problem: assign each customer to min(c_if + lambda_f)
        # adjusted_cost[i, f] = assign_cost[i, f] + lambda_f
        adjusted_cost = assign_cost + lambda_vec[np.newaxis, :]
        
        # For each customer, find best facility
        # Note: This relaxed solution ignores capacity constraints
        relaxed_assign = np.argmin(adjusted_cost, axis=1)
        
        # Calculate load
        load = np.zeros(n_facilities)
        np.add.at(load, relaxed_assign, demands)
        
        # 2. Update lambda (Subgradient method)
        # Gradient is (load - capacity)
        grad = load - capacities
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-6:
            break
            
        # Step size adjustment (simple diminishing step)
        # We want to push lambda up where load > cap, down where load < cap
        lambda_vec += step_size * grad / norm_grad
        
        # Keep lambda non-negative? Usually yes for capacity constraints >=
        lambda_vec = np.maximum(lambda_vec, 0)
        
        # 3. Project to a valid solution
        # We have a relaxed assignment. We need to fix capacity violations.
        # Strategy: 
        # - Identify facilities over capacity.
        # - Move customers from over-capacity facilities to under-capacity ones.
        # - Priority: Move customers with highest "penalty" (cost difference to next best)
        
        current_assign = relaxed_assign.copy()
        current_load = load.copy()
        
        # Repeat until valid
        for _ in range(50): # Max repair iterations
            over_facilities = np.where(current_load > capacities + 1e-9)[0]
            if len(over_facilities) == 0:
                break
                
            # Find a customer in an over-capacity facility to move
            # Pick the one with the largest "savings" by moving to the next best
            # Or simply the one that fits in the facility with most spare capacity
            moved = False
            for f_over in over_facilities:
                # Get customers assigned to f_over
                custs_in_f = np.where(current_assign == f_over)[0]
                if len(custs_in_f) == 0:
                    continue
                    
                # For each customer, find the best alternative facility
                best_alt_f = -1
                best_alt_cost_diff = -np.inf # We want to maximize (cost_to_current - cost_to_alt) ? 
                # Actually, we just want to find a feasible facility with min cost.
                # Let's just pick the customer with the smallest demand first (easiest to fit)
                # Or the one with the lowest cost to move.
                
                # Sort customers by demand ascending (easiest to move)
                custs_sorted = custs_in_f[np.argsort(demands[custs_in_f])]
                
                for i in custs_sorted:
                    d_i = demands[i]
                    # Find feasible facility with lowest assign cost
                    feasible = np.where(current_load + d_i <= capacities)[0]
                    # Exclude current facility
                    feasible = feasible[feasible != f_over]
                    if len(feasible) == 0:
                        continue
                        
                    # Pick the one with lowest assign cost
                    costs = assign_cost[i, feasible]
                    f_new = feasible[np.argmin(costs)]
                    
                    # Perform move
                    current_load[f_over] -= d_i
                    current_load[f_new] += d_i
                    current_assign[i] = f_new
                    moved = True
                    break # Move one customer and re-evaluate
                if moved:
                    break # Restart loop to re-check over-capacities
            
            if not moved:
                break # No more moves possible?
                
        # Check if valid
        if np.all(current_load <= capacities + 1e-9):
            cost = _compute_cost(current_assign, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = current_assign.copy()
                
        # Diminish step size
        step_size *= 0.95

    if best_result is not None:
        return best_result
    return _greedy_init(demands, capacities, open_cost, assign_cost)

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=10):
    """
    Local search with single moves and swaps.
    """
    n_customers = len(result)
    n_facilities = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        load[result[i]] += demands[i]
        
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result # Fallback to greedy if invalid
        
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Single Moves
        # For each customer, try moving to any other facility
        for i in range(n_customers):
            f_old = result[i]
            d_i = demands[i]
            
            # Find best facility to move to
            best_f = -1
            best_delta = 0.0 # We want delta < 0 (cost reduction)
            
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                if load[f_new] + d_i > capacities[f_new] + 1e-9:
                    continue
                
                # Calculate cost change
                # Change in assignment cost
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                
                # Change in open cost
                delta_open = 0.0
                if load[f_new] == 0:
                    # Opening new facility
                    delta_open += open_cost[f_new]
                if load[f_old] == d_i:
                    # Closing old facility
                    delta_open -= open_cost[f_old]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_f = f_new
                    
            if best_f != -1 and best_delta < -1e-9:
                # Perform move
                load[f_old] -= d_i
                load[best_f] += d_i
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        # 2. Swaps
        # For each pair of customers, try swapping their facilities
        # This is O(N^2). N=100 -> 5000 pairs. Fast enough.
        for i in range(n_customers):
            f_i = result[i]
            d_i = demands[i]
            for k in range(i + 1, n_customers):
                f_k = result[k]
                d_k = demands[k]
                
                if f_i == f_k:
                    continue
                
                # Check capacity constraints for swap
                # If f_i == f_k, no change in load.
                # If f_i != f_k:
                # Load at f_i becomes: load[f_i] - d_i + d_k
                # Load at f_k becomes: load[f_k] - d_k + d_i
                # Other facilities unchanged.
                
                new_load_i = load[f_i] - d_i + d_k
                new_load_k = load[f_k] - d_k + d_i
                
                if new_load_i > capacities[f_i] + 1e-9 or new_load_k > capacities[f_k] + 1e-9:
                    continue
                
                # Calculate cost change
                delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
                
                delta_open = 0.0
                # Check if facilities open/close
                # Facility f_i:
                old_load_i = load[f_i]
                # If old_load_i == d_i (only i was there), and new_load_i > 0 (k moves in), it stays open.
                # If old_load_i > d_i, it stays open.
                # It only closes if new_load_i == 0.
                if new_load_i == 0:
                    # f_i closes
                    delta_open -= open_cost[f_i]
                elif old_load_i == 0:
                    # f_i opens (impossible since i was there)
                    pass
                    
                # Facility f_k:
                old_load_k = load[f_k]
                if new_load_k == 0:
                    # f_k closes
                    delta_open -= open_cost[f_k]
                elif old_load_k == 0:
                    # f_k opens
                    delta_open += open_cost[f_k]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Perform swap
                    load[f_i] = new_load_i
                    load[f_k] = new_load_k
                    result[i] = f_k
                    result[k] = f_i
                    current_cost += total_delta
                    improved = True
                    break # Restart inner loop? Or just continue? 
                    # Restarting is safer to avoid invalid states from overlapping swaps
                    # But breaking inner loop and restarting outer is O(N^3) worst case.
                    # Given N=100, N^3=1e6, which is fine.
                    
    return result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # 1. Generate initial solutions
    # We will run Lagrangian init and Greedy init, pick the best one, then local search.
    
    sol1 = _greedy_init(demands, capacities, open_cost, assign_cost)
    cost1 = _compute_cost(sol1, demands, capacities, open_cost, assign_cost)
    
    sol2 = _lagrangian_init(demands, capacities, open_cost, assign_cost, steps=50)
    cost2 = _compute_cost(sol2, demands, capacities, open_cost, assign_cost)
    
    best_sol = sol1 if cost1 <= cost2 else sol2
    best_cost = min(cost1, cost2)
    
    # 2. Local Search
    # We have plenty of time. Let's run local search for a good amount of time.
    # If we have 120s, and 5 instances, we have ~24s per instance.
    # Local search is very fast. Let's give it 10s.
    
    improved_sol = _local_search(best_sol, demands, capacities, open_cost, assign_cost, time_limit=10)
    improved_cost = _compute_cost(improved_sol, demands, capacities, open_cost, assign_cost)
    
    if improved_cost < best_cost:
        best_sol = improved_sol
        best_cost = improved_cost
    
    # 3. If we have time left, do another round of perturbation + local search?
    # Let's try a simple random restart if time permits.
    if time.time() < deadline - 5:
        # Perturb the best solution: randomly reassign 10% of customers
        n_customers = len(demands)
        n_facilities = len(capacities)
        perturb_sol = best_sol.copy()
        num_perturb = max(1, n_customers // 10)
        indices = np.random.choice(n_customers, num_perturb, replace=False)
        
        # For each perturbed customer, assign to random feasible facility
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[perturb_sol[i]] += demands[i]
            
        for i in indices:
            f_old = perturb_sol[i]
            load[f_old] -= demands[i]
            
            # Find feasible facilities
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) > 0:
                f_new = feasible[np.random.randint(len(feasible))]
                perturb_sol[i] = f_new
                load[f_new] += demands[i]
        
        perturb_cost = _compute_cost(perturb_sol, demands, capacities, open_cost, assign_cost)
        
        if not np.isinf(perturb_cost):
            # Local search on perturbed solution
            perturbed_improved = _local_search(perturb_sol, demands, capacities, open_cost, assign_cost, time_limit=5)
            perturbed_improved_cost = _compute_cost(perturbed_improved, demands, capacities, open_cost, assign_cost)
            
            if perturbed_improved_cost < best_cost:
                best_sol = perturbed_improved
                best_cost = perturbed_improved_cost
    
    return best_sol
# EVOLVE-BLOCK-END
