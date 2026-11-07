"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 145.0  # Safety margin below 150s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for ease of use
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # Helper to check time
    def time_exceeded():
        return time.time() - start_time > timeout

    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum

    # --- Strategy 1: Mixed Integer Programming (Global Optimum) ---
    # We limit MIP time to allow more time for heuristic polishing if MIP doesn't solve quickly.
    
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.concatenate([assign_cost.flatten(), open_cost])
    
    # Integrality: all binary
    integrality = np.ones(n_vars)
    
    # Bounds: all [0, 1]
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Constraints
    # 1. Assignment: sum_j x_{ij} = 1 for each i
    A_eq = np.zeros((n_customers, n_vars))
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
    b_eq = np.ones(n_customers)
    
    # 2. Capacity: sum_i d_i * x_{ij} - C_j * y_j <= 0 for each j
    # Note: The current program uses this form. It is valid.
    # Alternatively, we can use x_ij <= y_j and sum d_i x_ij <= C_j.
    # The current form links them directly which is efficient.
    A_ub = np.zeros((n_facilities, n_vars))
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
    b_ub = np.zeros(n_facilities)
    
    constraint = LinearConstraint(A_ub, -np.inf, b_ub)
    constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)
    
    # Determine time limit for MIP solver
    # Give MIP 30 seconds. If it doesn't solve, we rely on heuristics with remaining time.
    mip_time_limit = 30.0
    
    best_result = None
    best_cost = float('inf')
    best_load = None
    
    try:
        # Solve MIP
        res = milp(
            c, 
            constraints=[constraint, constraint_eq], 
            integrality=integrality, 
            bounds=bounds, 
            options={"time_limit": mip_time_limit, "mip_rel_gap": 0.01}
        )
        
        if res.success and res.x is not None:
            # Extract solution
            x_flat = res.x[:n_customers * n_facilities]
            x_matrix = x_flat.reshape(n_customers, n_facilities)
            
            # Find assigned facility for each customer
            assignment = np.argmax(x_matrix, axis=1)
            
            # Verify validity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                f = assignment[i]
                load[f] += demands[i]
            
            valid = True
            for j in range(n_facilities):
                if load[j] > capacities[j] + 1e-5:
                    valid = False
                    break
            
            if valid:
                cost = calc_total_cost(assignment, load)
                best_result = assignment
                best_cost = cost
                best_load = load
                
    except Exception as e:
        pass

    # --- Strategy 2: Heuristic Fallback / Polish ---
    
    def greedy_init_largest_demand():
        """Greedy initialization: Largest Demand First"""
        order = np.argsort(-demands)
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                f = np.argmax(remaining_cap)
            else:
                costs = np.where(feasible_mask, assign_cost[i], np.inf)
                f = np.argmin(costs)
            
            res[i] = f
            load[f] += d
        
        return res, load

    def local_search(result, load):
        """
        Local search with 1-move (Relocation) and 2-move (Swap) improvements.
        Uses First-Improvement for Relocation (faster) and Best-Improvement for Swap.
        """
        current_cost = calc_total_cost(result, load)
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()
        
        improved = True
        max_iterations = 2000 
        iter_count = 0
        
        while improved and not time_exceeded() and iter_count < max_iterations:
            iter_count += 1
            improved = False
            
            # --- Phase A: Check 1-moves (Relocation) - First Improvement ---
            # We iterate customers in random order to avoid bias
            for i in np.random.permutation(n_customers):
                if time_exceeded():
                    return best_result, best_load, best_cost
                
                f_old = result[i]
                d = demands[i]
                
                # Pre-compute costs for all potential new facilities
                remaining_cap = capacities - load
                feasible_mask = remaining_cap >= d
                feasible_mask[f_old] = False # Can't move to same facility
                
                if not np.any(feasible_mask):
                    continue
                    
                # Calculate delta for all feasible facilities at once
                # delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                # delta_open depends on load[f_new] and load[f_old]
                
                # We need to check each feasible f_new to see if it improves cost
                # To be fast, we can just check the best few candidates?
                # Or just loop through feasible ones (max 20)
                
                best_delta_move = 0.0
                best_f_new = -1
                
                for f_new in np.where(feasible_mask)[0]:
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    
                    if load[f_old] == d:
                        delta_open -= open_cost[f_old]
                    
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta_move:
                        best_delta_move = total_delta
                        best_f_new = f_new
                
                if best_f_new != -1:
                    # Apply move
                    load[f_old] -= d
                    load[f_new] += d
                    result[i] = f_new
                    current_cost += best_delta_move
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = result.copy()
                        best_load = load.copy()
                    improved = True
                    # First improvement: break inner loop? No, we are iterating i.
                    # We apply the move and continue to next customer.
            
            # --- Phase B: Check 2-moves (Swap) - Best Improvement ---
            best_delta_swap = 0.0
            best_swap_data = None
            
            # Optimization: Only check swaps if time allows
            if not time_exceeded():
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
                        
                        if total_delta < best_delta_swap:
                            best_delta_swap = total_delta
                            best_swap_data = (i, j, f_i, f_j)
            
            if best_swap_data is not None:
                i, j, f_i, f_j = best_swap_data
                d_i = demands[i]
                d_j = demands[j]
                
                load[f_i] -= d_i
                load[f_j] -= d_j
                load[f_i] += d_j
                load[f_j] += d_i
                result[i] = f_j
                result[j] = f_i
                
                current_cost += best_delta_swap
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
                improved = True
                
        return best_result, best_load, best_cost

    # Initialize best solution
    if best_result is None:
        res, load = greedy_init_largest_demand()
        best_result = res
        best_load = load
        best_cost = calc_total_cost(best_result, best_load)
    else:
        # Use MIP solution as starting point for heuristic
        res, load = best_result.copy(), best_load.copy()

    # Run Local Search on the best available solution
    res_ls, load_ls, cost_ls = local_search(res, load)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_result = res_ls
        best_load = load_ls

    # Random Restarts with Perturbation
    def perturb(result, load, rng, strength=10):
        """
        Perturb the solution by performing 'strength' random valid moves.
        """
        res = result.copy()
        load = load.copy()
        
        for _ in range(strength):
            if time_exceeded():
                break
            
            move_type = rng.integers(0, 2) # 0: relocate, 1: swap
            
            if move_type == 0:
                i = rng.integers(0, n_customers)
                f_old = res[i]
                d = demands[i]
                
                feasible = np.where((capacities - load) >= d)[0]
                feasible = feasible[feasible != f_old]
                
                if len(feasible) > 0:
                    f_new = feasible[rng.integers(0, len(feasible))]
                    load[f_old] -= d
                    load[f_new] += d
                    res[i] = f_new
            else:
                i = rng.integers(0, n_customers)
                j = rng.integers(0, n_customers)
                
                if i == j or res[i] == res[j]:
                    continue
                
                f_i = res[i]
                f_j = res[j]
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
                res[i] = f_j
                res[j] = f_i
                
        return res, load

    rng = np.random.default_rng(42)
    restart_count = 0
    max_restarts = 50 # Increased restarts since we have more time budget for heuristics
    
    while not time_exceeded() and restart_count < max_restarts:
        # Adaptive strength: start small, increase later
        strength = 5 + int(15 * (restart_count / max_restarts))
        
        res_pert, load_pert = perturb(best_result, best_load, rng, strength)
        
        # Use a shorter local search for restarts to allow more restarts
        # We can reuse the same local_search but it might be too slow.
        # Let's just run it. The time_exceeded check inside will stop it.
        res_ls, load_ls, cost_ls = local_search(res_pert, load_pert)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_result = res_ls
            best_load = load_ls
            
        restart_count += 1
        
    return best_result.astype(int)
# EVOLVE-BLOCK-END
