"""Program using Lagrangian Relaxation and Subgradient Optimization for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Initial Solution: Greedy (Largest demand first, cheapest feasible)
    def greedy_init():
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    res[i] = f
                    break
            else:
                res[i] = fac_order[0]
                load[fac_order[0]] += demands[i]
        return res

    best_assign = greedy_init()
    
    # Calculate cost and load for a given assignment
    def get_load(assignment):
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[assignment[i]] += demands[i]
        return load

    def calc_cost(assignment):
        load = get_load(assignment)
        if np.any(load > capacities + 1e-9):
            return np.inf
        cost = 0.0
        for f in range(n_fac):
            if load[f] > 0:
                cost += open_cost[f]
        cost += np.sum(assign_cost[np.arange(n_cust), assignment])
        return cost

    best_cost = calc_cost(best_assign)
    
    # 2. Lagrangian Relaxation
    # We relax the capacity constraints: sum(d_i * x_if) <= u_f
    # L(x, lambda) = sum_i sum_f (c_if * x_if) + sum_f lambda_f * (sum_i d_i * x_if - u_f) + sum_f F_f * y_f
    # For fixed lambda, the problem decomposes by customer.
    # For customer i, choose f to minimize: c_if + lambda_f * d_i
    # However, we also have the fixed cost F_f * y_f.
    # This is a standard Facility Location problem.
    # We can solve the relaxed problem by iterating or using a heuristic.
    # Actually, the relaxed problem is still coupled by y_f.
    # But we can use a simpler approach:
    # Treat lambda_f as a penalty for using facility f.
    # Effective cost for customer i to facility f: c_if + lambda_f * d_i
    # We assign each customer to the argmin of this effective cost.
    # Then we check capacities.
    # If capacity is exceeded, increase lambda_f.
    # If capacity is underutilized (and cost is high?), decrease lambda_f.
    # This is essentially subgradient method.
    
    # Initialize lambda
    lambda_ = np.zeros(n_fac)
    
    # Subgradient parameters
    step_size = 1.0
    max_iters = 50000
    current_lambda = np.zeros(n_fac)
    
    # To handle the fixed cost F_f, we need to be careful.
    # If a facility is opened, we pay F_f.
    # In the Lagrangian, we usually keep y_f in the problem.
    # If we ignore F_f in the subproblem and add it back in the evaluation,
    # the lambda updates might not account for the fixed cost trade-off well.
    
    # Alternative: Include fixed cost in the "effective cost" if we assume the facility is open.
    # But which facilities are open?
    # A common heuristic for Lagrangian FL:
    # 1. Solve subproblem: min sum_i min_f (c_if + lambda_f d_i) x_if + sum_f F_f y_f
    #    where y_f = 1 if any customer assigned to f.
    #    This is still hard because of y_f.
    
    # Let's use a simpler "Penalty Method" variant which is easier to implement robustly.
    # Minimize: Total_Cost + Sum_f max(0, Load_f - Cap_f) * Penalty
    # This is equivalent to Lagrangian with lambda >= 0.
    
    # We will iterate:
    # 1. Compute effective costs: C_eff[i, f] = assign_cost[i, f] + lambda_[f] * demands[i]
    # 2. Assign each customer to argmin_f C_eff[i, f]
    # 3. Check loads.
    # 4. Update lambda_:
    #    If Load_f > Cap_f: lambda_[f] += step_size * (Load_f - Cap_f)
    #    If Load_f < Cap_f: lambda_[f] = max(0, lambda_[f] - step_size * (Cap_f - Load_f) * decay)
    #    (We only decrease if we want to allow more usage, but since we minimize, 
    #     if we are under capacity, it means the penalty wasn't high enough or cost was low.
    #     Actually, if we are under capacity, the constraint is slack, so lambda should be 0 
    #     in the optimal dual solution. So we should decrease towards 0.)
    
    # However, this ignores the fixed opening cost in the subproblem decision.
    # If a facility has high fixed cost but low variable cost, the penalty method might
    # open it unnecessarily if lambda is low.
    # To incorporate fixed costs, we can add a term to the effective cost.
    # But fixed cost is shared.
    
    # Let's try a hybrid:
    # Use the penalty method to find a feasible assignment.
    # Then, use local search to improve it (swap/move) considering the TRUE cost (with fixed costs).
    # The penalty method helps find a feasible region quickly.
    
    # Initialize lambda to encourage spreading load? No, start at 0.
    # If start at 0, it's just greedy by assign_cost.
    
    # Let's run the penalty loop for a while, keeping track of the best feasible solution found.
    
    current_assign = best_assign.copy()
    current_load = get_load(current_assign)
    
    # We need to be careful with step size.
    # Max demand is 35. Max capacity violation could be large.
    # Let's normalize step size by total demand or something?
    # Or just use a fixed small step and many iterations.
    
    # Adaptive step size:
    # step = alpha * (UB - LB) / ||g||^2
    # We don't have a tight LB easily.
    # Let's use a simple decaying step.
    
    base_step = 0.1
    current_step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0: # Leave 5s for final check
            break
            
        # 1. Compute effective costs
        # C_eff[i, f] = assign_cost[i, f] + lambda_[f] * demands[i]
        # Vectorized: assign_cost + np.outer(demands, lambda_)
        c_eff = assign_cost + np.outer(demands, current_lambda)
        
        # 2. Assign each customer to argmin
        new_assign = np.argmin(c_eff, axis=1)
        
        # 3. Check loads
        new_load = np.zeros(n_fac)
        for i in range(n_cust):
            new_load[new_assign[i]] += demands[i]
            
        # Check if valid
        if np.all(new_load <= capacities + 1e-9):
            # Calculate true cost
            true_cost = calc_cost(new_assign)
            if true_cost < best_cost:
                best_cost = true_cost
                best_assign = new_assign.copy()
                
        # 4. Update lambda_
        # Subgradient: g_f = Load_f - Cap_f
        # If Load_f > Cap_f, increase lambda_f
        # If Load_f < Cap_f, decrease lambda_f (towards 0)
        
        grad = new_load - capacities
        
        # Update
        # We want to move in direction of grad to reduce violation.
        # lambda_new = max(0, lambda + step * grad)
        # If grad > 0 (violation), lambda increases.
        # If grad < 0 (slack), lambda decreases.
        
        current_lambda = current_lambda + current_step * grad
        current_lambda = np.maximum(0, current_lambda)
        
        # Decay step size
        current_step *= 0.999
        
        # Optional: If we have a feasible solution, we can try to "repair" infeasible ones
        # or use the feasible one as a base for local search.
        
    # After Lagrangian/Penalty phase, perform Local Search on the best feasible solution found
    # to fix any issues and improve cost.
    
    if time.time() < deadline - 2.0:
        # Local Search on best_assign
        current_assign = best_assign.copy()
        current_cost = best_cost
        load = get_load(current_assign)
        
        # Simple Local Search: Move one customer to another facility if it improves cost and is feasible
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try moving each customer
            # To speed up, we can randomize order or just loop
            for i in range(n_cust):
                if time.time() > deadline - 1.0:
                    break
                current_f = current_assign[i]
                current_load_f = load[current_f]
                
                # Try all other facilities
                for new_f in range(n_fac):
                    if new_f == current_f:
                        continue
                    
                    # Check capacity of new_f
                    if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                        continue
                    
                    # Calculate delta cost
                    # Old contribution: assign_cost[i, current_f] + (open_cost[current_f] if load[current_f] == demands[i] else 0)
                    # New contribution: assign_cost[i, new_f] + (open_cost[new_f] if load[new_f] == 0 else 0)
                    # Note: open cost changes only if facility becomes empty or was empty.
                    
                    old_open_cost = open_cost[current_f] if load[current_f] == demands[i] else 0.0
                    new_open_cost = open_cost[new_f] if load[new_f] == 0.0 else 0.0
                    
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, current_f]
                    delta_open = new_open_cost - old_open_cost
                    delta_total = delta_assign + delta_open
                    
                    if delta_total < -1e-9:
                        # Apply move
                        load[current_f] -= demands[i]
                        load[new_f] += demands[i]
                        current_assign[i] = new_f
                        current_cost += delta_total
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assign = current_assign.copy()
                        break # Restart loop for this customer? Or continue? 
                             # If we move, load changes, so other moves might be invalid.
                             # Breaking the inner loop is safer. We restart the outer loop.
                             # Actually, breaking outer loop is safer for correctness of 'load' state.
                             # Let's just break inner and let outer continue, but 'load' is updated.
                             # However, 'improved' is True, so we will loop again.
                             # It's fine to continue checking other customers with updated load.
                             # But we broke the inner loop.
        
    return best_assign

# EVOLVE-BLOCK-END
