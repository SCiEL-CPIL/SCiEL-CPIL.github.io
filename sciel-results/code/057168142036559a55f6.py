import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for speed
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer for 120s limit
    
    # 1. Lagrangian Relaxation to get a good initial assignment pattern
    # We relax the capacity constraints: sum(d_i * x_if) <= K_f
    # Multipliers lambda_f
    lambdas = np.zeros(n_facilities)
    
    # We will run subgradient optimization for a fixed number of iterations or time
    # 1000 iterations is very fast (100 customers * 20 facilities = 2000 ops per iter)
    max_iters = 2000
    
    best_lower_bound = -np.inf
    best_lambda = lambdas.copy()
    
    # Step size parameters
    alpha = 1.0
    alpha_max = 10.0
    alpha_min = 0.01
    step_reduction = 0.95
    
    # Precompute costs scaled by demand? No, the term is c_if + lambda_f * d_i
    # c_if is assign_cost[i,f]
    
    for it in range(max_iters):
        # Check time
        if time.time() - start_time > time_limit * 0.5:
            break
            
        # 1. Solve the decomposed problem
        # For each customer i, find f that minimizes assign_cost[i,f] + lambdas[f] * demands[i]
        # Shape: (n_customers, n_facilities)
        adjusted_costs = assign_cost + np.outer(demands, lambdas)
        
        # Find best facility for each customer
        best_facilities = np.argmin(adjusted_costs, axis=1)
        
        # Calculate loads
        loads = np.zeros(n_facilities)
        np.add.at(loads, best_facilities, demands)
        
        # 2. Update multipliers (Subgradient)
        # Gradient of L(lambda) is (K_f - Load_f) ? 
        # L(lambda) = sum_i min_f (c_if + lambda_f d_i) - sum_f lambda_f K_f
        # dL/dlambda_f = sum_{i: f_i=f} d_i - K_f = Load_f - K_f
        # We want to maximize L(lambda) for lower bound? No, we want to find a good lambda.
        # Standard subgradient ascent for maximization: lambda <- lambda + alpha * (Load - K)
        # Wait, if Load > K, we need to increase lambda to discourage assignment.
        # So gradient direction is (Load_f - K_f).
        
        gradient = loads - capacities
        
        # Update lambdas
        # Use a simple step size rule
        # alpha = (UB - LB) / (sum(gradient^2) + eps) is standard but we don't have UB yet easily.
        # Let's use a fixed decaying step size.
        lambdas = np.maximum(0, lambdas + alpha * gradient)
        
        # Reduce step size
        alpha *= step_reduction
        if alpha < alpha_min:
            alpha = alpha_min
            # Restart with larger alpha if stuck? 
            # For now, just keep going.
            
        # Track best lower bound (optional, but good for diagnostics)
        # LB = sum(min costs) - sum(lambda * K)
        min_costs = np.min(adjusted_costs, axis=1)
        current_lb = np.sum(min_costs) - np.dot(lambdas, capacities)
        if current_lb > best_lower_bound:
            best_lower_bound = current_lb
            best_lambda = lambdas.copy()
            # Store the best assignment pattern from the best lambda
            best_assignment_pattern = best_facilities.copy()
    
    # Use the best lambda found to generate a base assignment
    # Re-evaluate with best_lambda
    adjusted_costs_best = assign_cost + np.outer(demands, best_lambda)
    base_assignment = np.argmin(adjusted_costs_best, axis=1)
    
    # 2. Feasibility Repair
    # The base_assignment might violate capacities.
    # We need to move customers from overloaded facilities to feasible ones.
    
    def repair_assignment(assignment):
        # Make a copy
        assignment = assignment.copy()
        loads = np.zeros(n_facilities)
        np.add.at(loads, assignment, demands)
        
        max_repair_iters = 500
        for _ in range(max_repair_iters):
            if time.time() - start_time > time_limit * 0.8:
                break
                
            # Find overloaded facilities
            overloaded = np.where(loads > capacities)[0]
            if len(overloaded) == 0:
                break
                
            # Pick the most overloaded facility
            f_over = overloaded[np.argmax(loads[overloaded] - capacities[overloaded])]
            
            # Get customers assigned to f_over
            customers_in_f = np.where(assignment == f_over)[0]
            
            # Sort customers by "cost to move" or just pick one?
            # Let's pick the customer that has the cheapest alternative facility that is feasible.
            # Or, to be greedy, pick the customer with the smallest demand?
            # Better: pick the customer that minimizes the increase in total assignment cost.
            
            best_move_cust = -1
            best_move_to = -1
            min_cost_increase = np.inf
            
            for i in customers_in_f:
                # Find feasible facilities for i, excluding f_over
                # Feasible if loads[f] + demands[i] <= capacities[f]
                feasible = []
                for f in range(n_facilities):
                    if f == f_over:
                        continue
                    if loads[f] + demands[i] <= capacities[f]:
                        # Cost increase: c_if_new - c_if_old
                        # Note: if f is empty, we might incur open_cost, but for now just consider assign_cost
                        # We will handle open_cost in local search.
                        cost_inc = assign_cost[i, f] - assign_cost[i, f_over]
                        feasible.append((cost_inc, f))
                
                if not feasible:
                    continue
                
                # Pick the cheapest feasible move
                feasible.sort(key=lambda x: x[0])
                cost_inc, f_new = feasible[0]
                
                if cost_inc < min_cost_increase:
                    min_cost_increase = cost_inc
                    best_move_cust = i
                    best_move_to = f_new
            
            if best_move_cust == -1:
                # No feasible move found? This should not happen if total demand <= total capacity
                # If it does, we have a stuck state. Break to avoid infinite loop.
                break
                
            # Perform the move
            loads[f_over] -= demands[best_move_cust]
            loads[best_move_to] += demands[best_move_cust]
            assignment[best_move_cust] = best_move_to
            
        return assignment, loads

    current_assignment, current_loads = repair_assignment(base_assignment)
    
    # 3. Local Search
    # Improve the feasible solution by moving or swapping customers.
    
    def total_cost(assignment):
        # Sum of assign_cost + sum of open_cost for used facilities
        used_facilities = np.unique(assignment)
        open_sum = np.sum(open_cost[used_facilities])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        return open_sum + assign_sum

    best_assignment = current_assignment.copy()
    best_cost = total_cost(best_assignment)
    
    # Local Search Loop
    improved = True
    ls_iter = 0
    max_ls_iters = 5000
    
    while improved and ls_iter < max_ls_iters:
        improved = False
        ls_iter += 1
        
        if time.time() - start_time > time_limit * 0.95:
            break
            
        # Try moving each customer to a different facility
        # To speed up, we can randomize the order or just iterate
        for i in range(n_customers):
            if time.time() - start_time > time_limit * 0.95:
                break
                
            f_current = best_assignment[i]
            d_i = demands[i]
            
            # Try all other facilities
            # We need to check if moving i to f_new is better
            # Delta Cost = (c_i,f_new - c_i,f_current) + Delta_Open_Cost
            
            # Delta_Open_Cost depends on whether f_current becomes empty and f_new was empty
            # f_current becomes empty if it only had customer i?
            # We can precompute counts or just check
            
            # For efficiency, let's calculate the potential benefit
            # We need to know if f_current will be empty after moving i
            # and if f_new is currently empty.
            
            # Count customers per facility
            # This is O(N) if done naively. Let's maintain counts.
            # But in this loop, we are iterating. Let's just compute on the fly or precompute once per outer loop?
            # Precomputing counts for each iteration is O(N). Total O(N^2). 100^2 = 10k. Very fast.
            
            counts = np.bincount(best_assignment, minlength=n_facilities)
            
            for f_new in range(n_facilities):
                if f_new == f_current:
                    continue
                
                # Check capacity
                if current_loads[f_new] + d_i > capacities[f_new]:
                    continue
                
                # Calculate cost change
                c_old = assign_cost[i, f_current]
                c_new = assign_cost[i, f_new]
                delta_assign = c_new - c_old
                
                delta_open = 0.0
                
                # Check if f_current becomes empty
                if counts[f_current] == 1:
                    # We will close f_current, save open_cost[f_current]
                    delta_open -= open_cost[f_current]
                
                # Check if f_new is currently empty
                if counts[f_new] == 0:
                    # We will open f_new, pay open_cost[f_new]
                    delta_open += open_cost[f_new]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Accept move
                    best_assignment[i] = f_new
                    current_loads[f_current] -= d_i
                    current_loads[f_new] += d_i
                    best_cost += total_delta
                    improved = True
                    # Break inner loop to restart evaluation with new state?
                    # Or continue? Restarting is safer for local search stability.
                    break
            
            # If we moved, the counts and loads changed.
            # The 'improved' flag is set. We break the inner loop.
            # We should probably break the outer loop too and restart the scan?
            # Or just continue scanning? Continuing might miss better moves if state changes significantly.
            # Given N=100, a full pass is cheap. Let's break outer loop on improvement to restart.
            if improved:
                break
    
    return best_assignment
