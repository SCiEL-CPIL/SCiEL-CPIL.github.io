"""Capacitated facility location: Lagrangian Relaxation with Greedy Primal Heuristic."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation of capacity constraints.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for ~145s max to leave buffer
    deadline = time.time() + 145.0
    
    demands_arr = np.asarray(demands, dtype=float)
    capacities_arr = np.asarray(capacities, dtype=float)
    open_cost_arr = np.asarray(open_cost, dtype=float)
    assign_cost_arr = np.asarray(assign_cost, dtype=float)
    
    # Initial best solution via simple greedy
    def greedy_solve(lambda_vec):
        """
        Solve the relaxed problem for given lambda.
        Returns: (assignment, load, objective_value)
        objective_value is the Lagrangian dual bound (lower bound for minimization).
        """
        # For each customer i, cost to assign to facility j is:
        # c_ij + lambda_j * d_i
        # But we also have fixed cost y_j.
        # The Lagrangian function L(lambda) = sum_i min_j (c_ij + lambda_j * d_i) + sum_j y_j * max(0, lambda_j)
        # Wait, the standard relaxation of sum(d_i * x_ij) <= cap_j is:
        # L(lambda) = min { sum c_ij x_ij + sum y_j y_j + sum lambda_j (sum d_i x_ij - cap_j) | x, y binary }
        #           = -sum lambda_j cap_j + min { sum_i sum_j (c_ij + lambda_j d_i) x_ij + sum_j y_j y_j | x, y binary, sum_j x_ij = 1 }
        
        # For fixed lambda, the problem decomposes by customer and facility.
        # For each customer i, choose j to minimize c_ij + lambda_j * d_i.
        # Let best_j[i] = argmin_j (c_ij + lambda_j * d_i)
        # Let min_cost[i] = min_j (c_ij + lambda_j * d_i)
        
        # However, this doesn't account for y_j (open cost) correctly in the decomposition 
        # because y_j is shared. 
        # Actually, if we relax capacity, the problem is:
        # min sum c_ij x_ij + sum y_j y_j + sum lambda_j (sum d_i x_ij - cap_j)
        # = -sum lambda_j cap_j + sum_i [ min_j (c_ij + lambda_j d_i) x_ij ] + sum_j y_j y_j
        # This is still coupled by y_j.
        
        # A common heuristic approach for Lagrangian in CFL:
        # 1. Compute "adjusted cost" for assignment: c_ij + lambda_j * d_i
        # 2. For each customer, find the facility that minimizes this.
        # 3. This gives a tentative assignment.
        # 4. Check capacities. If violated, repair.
        
        # Better Primal Heuristic:
        # We want a feasible solution.
        # Sort customers by demand descending.
        # For each customer, assign to the facility with min (c_ij + lambda_j * d_i) 
        # subject to capacity.
        # If no capacity, assign to facility with min (c_ij + lambda_j * d_i) + penalty?
        # Or just min c_ij among feasible.
        
        # Let's use the lambda to guide the greedy choice.
        # adjusted_costs = assign_cost_arr + np.outer(demands_arr, lambda_vec)
        
        # Greedy assignment:
        load = np.zeros(n_fac)
        res = np.zeros(n_cust, dtype=int)
        
        # Order customers by demand descending (largest first to fill capacity efficiently)
        order = np.argsort(-demands_arr)
        
        for i in order:
            d_i = demands_arr[i]
            # Feasible facilities
            feasible = np.where(load + d_i <= capacities_arr)[0]
            
            if len(feasible) == 0:
                # Infeasible, should not happen if total cap >= total demand
                # Pick facility with most remaining capacity
                f = int(np.argmax(capacities_arr - load))
                res[i] = f
                load[f] += d_i
            else:
                # Cost for feasible facilities
                # Use lambda-adjusted cost to guide choice
                costs = assign_cost_arr[i, feasible] + lambda_vec[feasible] * d_i
                # Also consider open cost? 
                # If a facility is not yet open, we might want to penalize opening it 
                # if lambda is high? 
                # For now, just use adjusted assignment cost.
                
                # To incorporate open cost in the heuristic:
                # If load[f] == 0, we pay open_cost[f].
                # So effective cost = c_ij + lambda_j * d_i + (open_cost_j if load_j == 0 else 0)
                
                effective_costs = costs.copy()
                for k in range(len(feasible)):
                    f_idx = feasible[k]
                    if load[f_idx] == 0:
                        effective_costs[k] += open_cost_arr[f_idx]
                
                best_idx = np.argmin(effective_costs)
                f_chosen = feasible[best_idx]
                res[i] = f_chosen
                load[f_chosen] += d_i
        
        # Calculate true objective
        used = np.unique(res)
        true_obj = np.sum(open_cost_arr[used]) + np.sum(assign_cost_arr[np.arange(n_cust), res])
        
        return res, load, true_obj

    # Initialize lambda to 0
    lambda_vec = np.zeros(n_fac)
    
    # Initial solution
    best_res, best_load, best_obj = greedy_solve(lambda_vec)
    
    # Subgradient Method
    # Step size: alpha_k * (UB - LB) / ||g||^2
    # We don't have a good LB easily. We can use the Lagrangian dual value.
    # L(lambda) = -sum lambda_j cap_j + sum_i min_j (c_ij + lambda_j d_i) + sum_j y_j y_j (if we could solve exactly)
    # Since we use a heuristic, we just track the best feasible solution found (UB).
    
    # To get a lower bound, we can compute:
    # LB = sum_i min_j (c_ij + lambda_j d_i) - sum_j lambda_j cap_j
    # Note: This LB ignores open costs, so it's a valid lower bound for the problem 
    # if we assume y_j >= 0 and the relaxation is valid.
    # Actually, min_j (c_ij + lambda_j d_i) is the cost for customer i.
    # Summing gives the variable cost part.
    # The term -sum lambda_j cap_j is constant for fixed lambda.
    
    # We will use a diminishing step size.
    # Start with a large step, reduce over time.
    
    max_iter = 2000
    step_size_factor = 2.0
    
    for k in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Compute subgradient
        # g_j = sum_i d_i * x_ij - cap_j
        # where x_ij is 1 if customer i is assigned to j in the relaxed solution.
        # In our heuristic, x_ij is the assignment from greedy_solve.
        # But the subgradient should be based on the solution to the *relaxed* problem.
        # Our greedy_solve is a heuristic for the relaxed problem.
        # Let's use the assignment from greedy_solve to compute the subgradient.
        
        # Re-run greedy to get assignment (we already have best_res, but it might not be optimal for current lambda)
        # Actually, we should solve the relaxed problem for the current lambda to get the subgradient.
        # Let's create a function that returns the assignment that minimizes the Lagrangian 
        # (ignoring capacity, just min cost per customer).
        
        # Relaxed solution:
        # For each customer i, j_i = argmin_j (c_ij + lambda_j d_i)
        # This is O(N * F).
        
        adj_costs = assign_cost_arr + np.outer(demands_arr, lambda_vec)
        relaxed_assign = np.argmin(adj_costs, axis=1)
        
        # Compute subgradient
        g = np.zeros(n_fac)
        for i in range(n_cust):
            g[relaxed_assign[i]] += demands_arr[i]
        g -= capacities_arr
        
        # Compute Lower Bound
        min_costs = np.min(adj_costs, axis=1)
        lb = np.sum(min_costs) - np.sum(lambda_vec * capacities_arr)
        
        # If LB >= UB, we can stop (optimality gap closed)
        # But LB is often very loose.
        
        # Update lambda
        norm_g_sq = np.dot(g, g)
        if norm_g_sq < 1e-9:
            break
            
        step_size = step_size_factor * (best_obj - lb) / norm_g_sq
        # Clip step size to prevent divergence
        max_step = 10.0
        if step_size > max_step:
            step_size = max_step
        if step_size < 0:
            step_size = 0
            
        lambda_vec += step_size * g
        
        # Ensure lambda >= 0? 
        # In minimization with <= constraints, dual variables (lambda) should be >= 0.
        lambda_vec = np.maximum(lambda_vec, 0)
        
        # Get new feasible solution from heuristic
        new_res, new_load, new_obj = greedy_solve(lambda_vec)
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_res = new_res
            best_load = new_load
            # Reset step size factor to explore more
            step_size_factor = 2.0
        else:
            # Decay step size
            step_size_factor *= 0.95
            
        # Check time
        if time.time() > deadline - 10.0:
            break
            
    # Final local search polish on the best solution found
    def local_search(res, load):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            any_improve = False
            base_assign = assign_cost_arr[np.arange(n_cust), res]
            
            # Single moves
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                f_old = res[i]
                d_i = demands_arr[i]
                feasible_mask = (load + d_i <= capacities_arr)
                feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
                if not np.any(feasible_mask):
                    continue
                delta_assign = assign_cost_arr[i, :] - base_assign[i]
                delta_open = np.zeros(n_fac)
                if load[f_old] == d_i:
                    delta_open[f_old] = -open_cost_arr[f_old]
                open_new = (load == 0) & (np.arange(n_fac) != f_old)
                if np.any(open_new):
                    delta_open[open_new] = open_cost_arr[open_new]
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf
                total_delta[f_old] = np.inf
                best_f_new = int(np.argmin(total_delta))
                best_delta = total_delta[best_f_new]
                if best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    any_improve = True
            if any_improve:
                improved = True
                # Continue loop to find more improvements
            else:
                break
                
        return res, load

    best_res, best_load = local_search(best_res.copy(), best_load.copy())
    
    # Verify capacity
    if np.any(best_load > capacities_arr + 1e-6):
        # Fallback to simple greedy if something went wrong
        res_greedy = np.zeros(n_cust, dtype=int)
        load_greedy = np.zeros(n_fac)
        order = np.argsort(-demands_arr)
        for i in order:
            feasible = np.where(load_greedy + demands_arr[i] <= capacities_arr)[0]
            if len(feasible) == 0:
                f = int(np.argmax(capacities_arr - load_greedy))
            else:
                costs = assign_cost_arr[i, feasible]
                f = int(feasible[np.argmin(costs)])
            res_greedy[i] = f
            load_greedy[f] += demands_arr[i]
        best_res = res_greedy

    return best_res
# EVOLVE-BLOCK-END
