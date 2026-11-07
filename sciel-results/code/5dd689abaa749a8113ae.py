import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Lagrangian relaxation: dual variables lambda_f for capacity constraints
    # Primal: min sum_i,f c[i,f]*x[i,f] + sum_f y_f * open_cost[f]
    #         s.t. sum_f x[i,f] = 1 for all i
    #              sum_i d[i]*x[i,f] <= cap[f] for all f
    #              x[i,f] in {0,1}, y_f in {0,1}
    #
    # Lagrangian: L(x,y,lambda) = sum_i,f (c[i,f] + lambda_f*d[i])*x[i,f] + sum_f y_f*open_cost[f]
    #   - sum_f lambda_f*cap[f]
    #
    # For fixed lambda, the problem decomposes:
    #   For each customer i: pick f that minimizes c[i,f] + lambda_f*d[i]
    #   For each facility f: decide y_f to minimize y_f*open_cost[f] subject to
    #     if sum_i d[i]*x[i,f] > cap[f], we need y_f=1 (but x is already assigned)
    #
    # Actually, let's use a simpler approach:
    # For fixed lambda, assign each customer to argmin_f (c[i,f] + lambda_f*d[i])
    # Then check capacity violations. If facility f is over capacity, increase lambda_f.
    # If under capacity, decrease lambda_f.
    # After convergence, we have a good assignment. Then do local search.
    
    # Initialize lambda to zeros
    lambda_f = np.zeros(n_facilities)
    
    # Track best solution
    best_assignment = None
    best_cost = float('inf')
    
    # Subgradient method parameters
    max_iter = 2000
    step_size = 1.0
    best_step = 1.0
    
    for iteration in range(max_iter):
        if time.time() > deadline - 30.0:
            break
        
        # For fixed lambda, assign each customer to min over f of (c[i,f] + lambda_f*d[i])
        # This is: for each i, argmin_f (assign_cost[i,f] + lambda_f * demands[i])
        
        # Compute adjusted costs: shape (n_customers, n_facilities)
        adjusted = assign_cost + np.outer(demands, lambda_f)
        
        # Assign each customer to argmin
        assignment = np.argmin(adjusted, axis=1)
        
        # Check capacity
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        # Compute subgradient: g_f = load[f] - cap[f]
        subgrad = load - capacities
        
        # Compute Lagrangian value (lower bound)
        lagrangian_val = np.sum(adjusted[np.arange(n_customers), assignment]) - np.dot(lambda_f, capacities)
        
        # Update lambda using subgradient
        # Step size: use Polyak step if we have an upper bound, else diminishing
        if best_cost < float('inf'):
            # Polyak step: step = (UB - LB) / ||g||^2
            lb = lagrangian_val
            ub = best_cost
            if ub > lb:
                step = (ub - lb) / (np.dot(subgrad, subgrad) + 1e-10)
                step = min(step, best_step)  # Cap step size
            else:
                step = step_size / (iteration + 1)
        else:
            step = step_size / (iteration + 1)
        
        # Update lambda: lambda_f += step * subgrad_f, project to non-negative
        lambda_f = np.maximum(0, lambda_f + step * subgrad)
        
        # Check if this assignment is feasible
        if np.all(load <= capacities + 1e-9):
            # Compute actual cost
            open_facilities = np.unique(assignment)
            cost = np.sum(open_cost[open_facilities])
            for i in range(n_customers):
                cost += assign_cost[i, assignment[i]]
            
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment.copy()
                best_step = min(step, 1.0)  # Keep best step
        
        # Decay step size
        step_size *= 0.995
    
    # If no feasible solution found, use greedy initialization
    if best_assignment is None:
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        
        for i in order:
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                f = np.argmax(capacities - load)
            else:
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            
            assignment[i] = f
            load[f] += demands[i]
        
        best_assignment = assignment
        best_cost = float('inf')
        open_facilities = np.unique(assignment)
        best_cost = np.sum(open_cost[open_facilities])
        for i in range(n_customers):
            best_cost += assign_cost[i, assignment[i]]
    
    # Local search on the best feasible solution
    assignment = best_assignment.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[assignment[i]] += demands[i]
    
    current_cost = best_cost
    
    # Simulated annealing / local search
    T = 10.0
    cooling = 0.95
    max_ls_iter = 5000
    
    for ls_iter in range(max_ls_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Pick a random customer
        i = np.random.randint(n_customers)
        current_f = assignment[i]
        
        # Try moving to a different facility
        # Pick a random other facility, or the best one
        other_facilities = np.setdiff1d(np.arange(n_facilities), current_f)
        if len(other_facilities) == 0:
            continue
        
        # With some probability, pick the best feasible facility
        if np.random.random() < 0.5:
            feasible = other_facilities[load[other_facilities] + demands[i] <= capacities[other_facilities]]
            if len(feasible) == 0:
                # Pick random
                f = other_facilities[np.random.randint(len(other_facilities))]
            else:
                f = feasible[np.argmin(assign_cost[i, feasible])]
        else:
            f = other_facilities[np.random.randint(len(other_facilities))]
        
        # Check feasibility
        if load[f] + demands[i] > capacities[f]:
            continue
        
        # Compute delta
        delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
        
        # Check if current_f becomes empty (saves open cost)
        if load[current_f] - demands[i] <= 0:
            delta_open = -open_cost[current_f]
        else:
            delta_open = 0
        
        # Check if f was empty (adds open cost)
        if load[f] <= 0:
            delta_open += open_cost[f]
        
        delta = delta_assign + delta_open
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Make the move
            load[current_f] -= demands[i]
            load[f] += demands[i]
            assignment[i] = f
            current_cost += delta
        
        T *= cooling
    
    # Final local search: greedy improvement
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        for i in range(n_customers):
            current_f = assignment[i]
            open_set = set(np.unique(assignment))
            
            best_cost_i = assign_cost[i, current_f]
            best_f = current_f
            
            for f in open_set:
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    if assign_cost[i, f] < best_cost_i:
                        best_cost_i = assign_cost[i, f]
                        best_f = f
            
            if best_f != current_f:
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
                assignment[i] = best_f
                current_cost += (assign_cost[i, best_f] - assign_cost[i, current_f])
                improved = True
        
        # Try to close facilities with 1 customer
        open_set = set(np.unique(assignment))
        for f in list(open_set):
            customers_in_f = np.where(assignment == f)[0]
            if len(customers_in_f) == 1:
                i = customers_in_f[0]
                other_open = [g for g in open_set if g != f]
                if not other_open:
                    continue
                
                best_cost_i = float('inf')
                best_g = -1
                for g in other_open:
                    if load[g] + demands[i] <= capacities[g]:
                        if assign_cost[i, g] < best_cost_i:
                            best_cost_i = assign_cost[i, g]
                            best_g = g
                
                if best_g != -1:
                    delta = -open_cost[f] + (assign_cost[i, best_g] - assign_cost[i, f])
                    if delta < 0:
                        load[f] -= demands[i]
                        load[best_g] += demands[i]
                        assignment[i] = best_g
                        current_cost += delta
                        improved = True
                        break
    
    return assignment
