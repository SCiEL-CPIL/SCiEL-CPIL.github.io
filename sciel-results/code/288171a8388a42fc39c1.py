import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds, linprog
from scipy.optimize import linear_sum_assignment

def _greedy_init(demands, capacities, assign_cost):
    n_c = len(demands)
    n_f = len(capacities)
    order = np.argsort(-demands)
    result = np.full(n_c, -1, dtype=int)
    load = np.zeros(n_f)
    
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            for f in range(n_f):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    return result

def _local_search_polish(assignment, demands, capacities, assign_cost, open_cost, deadline):
    n_c = len(demands)
    n_f = len(capacities)
    
    # Current cost
    def get_cost(assignment):
        total = 0.0
        load = np.zeros(n_f)
        for i in range(n_c):
            f = assignment[i]
            total += assign_cost[i, f]
            load[f] += demands[i]
        for f in range(n_f):
            if load[f] > 0:
                total += open_cost[f]
        return total, load

    current_cost, current_load = get_cost(assignment)
    best_cost = current_cost
    best_assignment = assignment.copy()
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try moving each customer to a different facility
        for i in range(n_c):
            if (time.time() - start_time) > deadline:
                break
            current_f = assignment[i]
            # Try all other facilities
            for new_f in range(n_f):
                if new_f == current_f:
                    continue
                # Check capacity
                if current_load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Calculate new cost
                # Delta assignment cost
                delta_assign = assign_cost[i, new_f] - assign_cost[i, current_f]
                
                # Delta open cost
                delta_open = 0.0
                # If new_f was empty, we open it
                if current_load[new_f] == 0:
                    delta_open += open_cost[new_f]
                # If current_f becomes empty, we close it
                if current_load[current_f] == demands[i]:
                    delta_open -= open_cost[current_f]
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    # Accept move
                    assignment[i] = new_f
                    current_load[current_f] -= demands[i]
                    current_load[new_f] += demands[i]
                    current_cost += total_delta
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                    improved = True
                    break # Restart loop for this customer? Or just continue?
                    # Let's just continue to next customer for simplicity, 
                    # but we updated the state.
    
    return best_assignment, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = 145.0 # Safety margin from 150s
    
    # 1. Greedy Initialization
    initial_assign = _greedy_init(demands, capacities, assign_cost)
    
    # 2. MIP Formulation
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_customers * n_facilities + f] = open_cost[f]
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality
    integrality = np.ones(n_vars)
    
    # Constraints Matrix A
    n_rows = n_customers + n_facilities
    A = np.zeros((n_rows, n_vars))
    lb_constr = np.zeros(n_rows)
    ub_constr = np.zeros(n_rows)
    
    # Fill Constraint 1 (Customer assignment)
    for i in range(n_customers):
        row_idx = i
        for f in range(n_facilities):
            col_idx = i * n_facilities + f
            A[row_idx, col_idx] = 1.0
        lb_constr[row_idx] = 1.0
        ub_constr[row_idx] = 1.0
        
    # Fill Constraint 2 (Capacity)
    for f in range(n_facilities):
        row_idx = n_customers + f
        for i in range(n_customers):
            col_idx = i * n_facilities + f
            A[row_idx, col_idx] = demands[i]
        col_idx_y = n_customers * n_facilities + f
        A[row_idx, col_idx_y] = -capacities[f]
        lb_constr[row_idx] = -np.inf
        ub_constr[row_idx] = 0.0
        
    constraints = LinearConstraint(A, lb_constr, ub_constr)
    
    best_solution = initial_assign
    best_cost = None
    
    # Try to solve MIP with remaining time
    remaining_time = deadline - (time.time() - start_time) - 5.0
    if remaining_time > 5.0:
        try:
            res = milp(c=c,
                       constraints=constraints,
                       integrality=integrality,
                       bounds=bounds,
                       options={'time_limit': remaining_time, 'mip_rel_gap': 0.001})
            
            if res.status in [0, 1]: # Optimal or feasible
                x_flat = res.x
                x = x_flat[:n_customers * n_facilities].reshape(n_customers, n_facilities)
                assignment = np.argmax(x, axis=1)
                
                # Verify feasibility
                load = np.zeros(n_facilities)
                for i in range(n_customers):
                    load[assignment[i]] += demands[i]
                
                if np.all(load <= capacities + 1e-6):
                    # Calculate cost to compare
                    cost = 0.0
                    for f in range(n_facilities):
                        if load[f] > 0:
                            cost += open_cost[f]
                    for i in range(n_customers):
                        cost += assign_cost[i, assignment[i]]
                    
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_solution = assignment
                        
        except Exception:
            pass
    
    # 3. Local Search Polish
    # Use remaining time for local search
    remaining_time = deadline - (time.time() - start_time) - 2.0
    if remaining_time > 1.0:
        polished_assign, polished_cost = _local_search_polish(best_solution, demands, capacities, assign_cost, open_cost, remaining_time)
        
        # Verify feasibility of polished solution
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[polished_assign[i]] += demands[i]
        
        if np.all(load <= capacities + 1e-6):
            # Recalculate cost to be sure
            cost = 0.0
            for f in range(n_facilities):
                if load[f] > 0:
                    cost += open_cost[f]
            for i in range(n_customers):
                cost += assign_cost[i, polished_assign[i]]
                
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_solution = polished_assign
            else:
                # If local search didn't improve, stick with best found
                pass
        else:
            # If local search broke feasibility (shouldn't happen), revert
            pass

    return best_solution

    # 2. MIP Formulation
    # Variables:
    # x[i, f] in {0, 1} : 1 if customer i is assigned to facility f
    # y[f] in {0, 1} : 1 if facility f is open
    
    # Objective:
    # Minimize sum_{i,f} assign_cost[i,f] * x[i,f] + sum_f open_cost[f] * y[f]
    
    # Constraints:
    # 1. sum_f x[i,f] = 1 for all i (Each customer assigned to exactly one facility)
    # 2. sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f (Capacity constraint)
    #    Note: This implies y[f] must be 1 if any customer is assigned to f.
    #    If y[f]=0, then sum_i demands[i] x[i,f] <= 0, which means x[i,f]=0 for all i.
    #    So this single constraint handles both capacity and open/close logic.
    
    # Flatten x and y into a single vector for scipy.optimize.milp
    # Order: x[0,0], x[0,1], ..., x[0,F-1], x[1,0], ..., x[n_c-1, F-1], y[0], ..., y[F-1]
    
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_customers * n_facilities + f] = open_cost[f]
    
    # Bounds
    # All variables are binary, so bounds are [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality
    integrality = np.ones(n_vars)
    
    # Constraints Matrix A
    # We need to build A such that lb_constr <= A @ vars <= ub_constr
    
    # Constraint 1: sum_f x[i,f] = 1 for each i
    # There are n_customers such constraints.
    # Row for customer i:
    # x[i,0] + ... + x[i, F-1] = 1
    # In the flattened vector, x[i,f] is at index i * n_facilities + f.
    
    # Constraint 2: sum_i demands[i] x[i,f] <= capacities[f] * y[f] for each f
    # Rearranged: sum_i demands[i] x[i,f] - capacities[f] * y[f] <= 0
    # There are n_facilities such constraints.
    # Row for facility f:
    # sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
    # Indices:
    # x[i,f] is at i * n_facilities + f.
    # y[f] is at n_customers * n_facilities + f.
    
    # Total rows: n_customers + n_facilities
    n_rows = n_customers + n_facilities
    A = np.zeros((n_rows, n_vars))
    lb_constr = np.zeros(n_rows)
    ub_constr = np.zeros(n_rows)
    
    # Fill Constraint 1 (Customer assignment)
    for i in range(n_customers):
        row_idx = i
        for f in range(n_facilities):
            col_idx = i * n_facilities + f
            A[row_idx, col_idx] = 1.0
        lb_constr[row_idx] = 1.0
        ub_constr[row_idx] = 1.0
        
    # Fill Constraint 2 (Capacity)
    for f in range(n_facilities):
        row_idx = n_customers + f
        for i in range(n_customers):
            col_idx = i * n_facilities + f
            A[row_idx, col_idx] = demands[i]
        # Coefficient for y[f]
        col_idx_y = n_customers * n_facilities + f
        A[row_idx, col_idx_y] = -capacities[f]
        lb_constr[row_idx] = -np.inf
        ub_constr[row_idx] = 0.0
        
    constraints = LinearConstraint(A, lb_constr, ub_constr)
    
    # Solve MIP
    # Use the Lagrangian result as a warm start? 
    # scipy.optimize.milp does not directly accept a warm start solution in the same way,
    # but we can check if the Lagrangian solution is good enough or just let HiGHS do its thing.
    # Given the small size (100x20 = 2000 vars + 20 = 2020 vars, 120 constraints),
    # HiGHS should be fast.
    
    start_time = time.time()
    deadline = 140.0
    
    try:
        res = milp(c=c,
                   constraints=constraints,
                   integrality=integrality,
                   bounds=bounds,
                   options={'time_limit': deadline - start_time - 5.0, 'mip_rel_gap': 0.001})
        
        if res.status == 0: # Optimal or feasible
            x_flat = res.x
            # Extract x
            x = x_flat[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            
            # Assign each customer to the facility with x[i,f] closest to 1
            # Since it's a MIP, x should be 0 or 1.
            # If it's relaxed (LP relaxation), we might need rounding.
            # But milp with integrality=1 returns integer solutions.
            
            assignment = np.argmax(x, axis=1)
            
            # Verify feasibility
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignment[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return assignment
            else:
                # If MIP returns infeasible or invalid, fall back to Lagrangian/Greedy
                pass
                
    except Exception:
        # If MIP fails, fall back
        pass
        
    # Fallback: Use Lagrangian initialized solution
    return lagrangian_init(demands, capacities, open_cost, assign_cost)
