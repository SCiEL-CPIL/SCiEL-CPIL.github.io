import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = None
    best_score = float('inf')
    
    def evaluate(result):
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
        
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                f = np.argmin(assign_cost[i])
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
            
            # 1. Try 1-moves (relocation)
            for i in range(n_customers):
                f_old = result[i]
                d = demands[i]
                
                best_delta = 1e-9
                best_f_new = -1
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    if load[f_new] + d > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    if load[f_old] - d == 0:
                        delta_open -= open_cost[f_old]
                        
                    if load[f_new] == 0:
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

            # 2. Try 2-swaps (interchange)
            for i in range(n_customers):
                if time.time() > deadline:
                    return result, load, is_open
                for j in range(i + 1, n_customers):
                    f_i = result[i]
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    old_open_i = is_open[f_i]
                    old_open_j = is_open[f_j]
                    
                    new_open_i = (new_load_i > 0)
                    new_open_j = (new_load_j > 0)
                    
                    delta_open = 0.0
                    if old_open_i and not new_open_i:
                        delta_open -= open_cost[f_i]
                    elif not old_open_i and new_open_i:
                        delta_open += open_cost[f_i]
                        
                    if old_open_j and not new_open_j:
                        delta_open -= open_cost[f_j]
                    elif not old_open_j and new_open_j:
                        delta_open += open_cost[f_j]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        result[i] = f_j
                        result[j] = f_i
                        
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        
                        if new_load_i == 0:
                            is_open[f_i] = False
                        else:
                            is_open[f_i] = True
                            
                        if new_load_j == 0:
                            is_open[f_j] = False
                        else:
                            is_open[f_j] = True
                            
                        improved = True
                        
                        if time.time() > deadline:
                            return result, load, is_open
                            
        return result, load, is_open

    def lagrangian_relaxation():
        # Solve the Lagrangian relaxation of the capacitated facility location problem.
        # Multipliers lambda_f for capacity constraints.
        # Subproblem: min sum_i sum_f (c_if + lambda_f * x_if) + sum_f F_f * y_f
        # s.t. sum_f x_if = 1, x_if <= y_f, y_f in {0,1}
        
        # We use a subgradient method to update lambda.
        # Initial lambda = 0
        lambda_mult = np.zeros(n_facilities)
        step_size = 1.0
        best_primal = float('inf')
        best_solution = None
        
        # Upper bound for lambda (if lambda > max_assign_cost, it's never beneficial to assign to that facility)
        max_cost = np.max(assign_cost)
        
        for it in range(500):
            if time.time() > deadline - 10:
                break
            
            # Solve subproblem for current lambda
            # cost_if_lambda = assign_cost[i, f] + lambda_mult[f]
            cost_lambda = assign_cost + lambda_mult[np.newaxis, :]
            
            # For each customer i, the best facility is argmin_f (cost_lambda[i, f] + open_cost[f] if y_f=1 else inf)
            # But y_f is shared. This is a fixed-charge assignment problem.
            # Actually, for a fixed lambda, the problem is still coupled by y_f.
            # Wait, the Lagrangian relaxation removes the capacity constraint.
            # The remaining problem is: min sum_i sum_f (c_if + lambda_f x_if) + sum_f F_f y_f
            # s.t. sum_f x_if = 1, x_if <= y_f.
            # This is a standard Uncapacitated Facility Location Problem (UFLP).
            # It can be solved by local search or other methods, but it's still NP-hard.
            # However, a simple greedy or local search on UFLP is fast.
            
            # Let's use a simple local search for UFLP.
            # Or, we can approximate the dual function value.
            # A lower bound on the subproblem value is sum_i min_f (c_if + lambda_f).
            # This lower bound is valid because y_f >= x_if implies y_f >= 0, and F_f >= 0.
            # Actually, if min_f (c_if + lambda_f) is achieved by f*, then y_f* must be 1 if x_if* = 1.
            # So the cost is at least min_f (c_if + lambda_f) + F_f* (if f* was closed).
            
            # Let's just use the lower bound for the subgradient direction.
            # g_f = sum_i x_if - capacity_f
            # But we don't have x_if explicitly without solving UFLP.
            
            # Alternative: Use the "price of assignment" approach.
            # Assign each customer to the facility that minimizes c_if + lambda_f.
            # This ignores the open cost, which is a lower bound on the true subproblem cost.
            
            assignment = np.argmin(cost_lambda, axis=1)
            load = np.zeros(n_facilities)
            for i, f in enumerate(assignment):
                load[f] += demands[i]
            
            # Subgradient: g_f = load_f - capacity_f
            g = load - capacities
            
            # Update lambda
            # lambda_f = max(0, lambda_f + step_size * g_f)
            lambda_mult = np.maximum(0, lambda_mult + step_size * g)
            
            # Decrease step size
            step_size *= 0.95
            
            # Evaluate the primal solution (assignment) if it's feasible
            if np.all(load <= capacities + 1e-9):
                score = evaluate(assignment)
                if score < best_primal:
                    best_primal = score
                    best_solution = assignment.copy()
            
            # Also try to repair infeasible solutions using local search
            # This is expensive, so we do it periodically
            if it % 10 == 0 and best_solution is not None:
                # We could run local search on best_solution, but let's stick to the main loop structure.
                pass
            
            # If we have a feasible solution, we can also try to improve it with local search
            # But that's what the main loop does.
            
        return best_solution

    # Main loop
    # 1. Deterministic greedy
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()
        
    # 2. Lagrangian Relaxation
    if time.time() < deadline - 20:
        lag_sol = lagrangian_relaxation()
        if lag_sol is not None:
            lag_load = np.zeros(n_facilities)
            lag_open = np.zeros(n_facilities, dtype=bool)
            for i, f in enumerate(lag_sol):
                lag_load[f] += demands[i]
                lag_open[f] = True
            lag_sol, lag_load, lag_open = local_search(lag_sol, lag_load, lag_open)
            lag_score = evaluate(lag_sol)
            if lag_score < best_score:
                best_score = lag_score
                best_result = lag_sol.copy()

    # 3. Randomized restarts
    for k in range(50):
        if time.time() > deadline - 5:
            break
            
        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = local_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
            
    return best_result
