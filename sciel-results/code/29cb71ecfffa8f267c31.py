"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
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

    # --- Strategy 1: Mixed Integer Programming (Global Optimum) ---
    # Formulate as MIP
    # Variables: x_{ij} (N*M) and y_j (M)
    # Total vars: N*M + M
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    # c = [assign_cost flattened, open_cost]
    c = np.concatenate([assign_cost.flatten(), open_cost])
    
    # Integrality: all binary
    integrality = np.ones(n_vars)
    
    # Bounds: all [0, 1]
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Constraints
    # 1. Assignment: sum_j x_{ij} = 1 for each i
    # A_eq shape: (N, N*M + M)
    A_eq = np.zeros((n_customers, n_vars))
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
    b_eq = np.ones(n_customers)
    
    # 2. Capacity: sum_i d_i * x_{ij} - C_j * y_j <= 0 for each j
    # A_ub shape: (M, N*M + M)
    A_ub = np.zeros((n_facilities, n_vars))
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        # Coefficient for y_j
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
    b_ub = np.zeros(n_facilities)
    
    constraint = LinearConstraint(A_ub, -np.inf, b_ub)
    constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)
    
    try:
        # Solve MIP
        # options={'time_limit': 10} # Optional, but milp handles time limit via options
        res = milp(c, constraints=[constraint, constraint_eq], integrality=integrality, bounds=bounds, options={"time_limit": 10})
        
        if res.success and res.x is not None:
            # Extract solution
            x_flat = res.x[:n_customers * n_facilities]
            # y = res.x[n_customers * n_facilities:]
            
            # Convert to assignment array
            # x_flat is flattened row-major: x[0,0], x[0,1]... x[0,M-1], x[1,0]...
            x_matrix = x_flat.reshape(n_customers, n_facilities)
            
            # Find assigned facility for each customer
            # Due to floating point, we might have 0.999 or 1.001
            # argmax should work if one is clearly 1
            assignment = np.argmax(x_matrix, axis=1)
            
            # Verify validity
            # Check if all customers assigned to a facility with x > 0.5
            # Check capacities
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
                return assignment.astype(int)
                
    except Exception as e:
        # If MIP fails, fall through to heuristic
        pass

    # --- Strategy 2: Heuristic Fallback (Current Program Logic) ---
    
    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum

    def greedy_init():
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
        """Local search with 1-move and 2-move (swap) improvements"""
        current_cost = calc_total_cost(result, load)
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()
        
        improved = True
        while improved:
            if time_exceeded():
                break
            
            improved = False
            best_move_type = None
            best_move_data = None
            best_delta = 0.0
            
            # --- Phase A: Check 1-moves (Relocation) ---
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
            
            # --- Phase B: Check 2-moves (Swap) ---
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
                
                if time_exceeded():
                    break
            else:
                break
        
        return best_result, best_load, best_cost

    # 1. Initial greedy solution
    result, load = greedy_init()
    best_result, best_load, best_cost = local_search(result, load)
    
    # 2. Multiple random restarts if time permits
    rng = np.random.default_rng(42)
    restart_count = 0
    max_restarts = 50  # Cap on number of restarts
    
    while not time_exceeded() and restart_count < max_restarts:
        # Random perturbation of current best solution
        result = best_result.copy()
        load = best_load.copy()
        
        # Perform 5-10 random swaps/relocations
        num_perturbations = rng.integers(5, 11)
        for _ in range(num_perturbations):
            if time_exceeded():
                break
            
            move_type = rng.integers(0, 2)  # 0 = relocation, 1 = swap
            
            if move_type == 0:
                # Relocation
                i = rng.integers(0, n_customers)
                f_old = result[i]
                d = demands[i]
                
                # Try to find a feasible new facility
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
