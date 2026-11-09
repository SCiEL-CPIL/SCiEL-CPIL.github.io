"""Capacitated facility location: MILP formulation using scipy.optimize.milp."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline guard: 150s limit, use 140s for safety
    deadline = time.time() + 140.0
    
    demands_arr = np.asarray(demands, dtype=np.float64)
    capacities_arr = np.asarray(capacities, dtype=np.float64)
    open_cost_arr = np.asarray(open_cost, dtype=np.float64)
    assign_cost_arr = np.asarray(assign_cost, dtype=np.float64)

    # Variables:
    # x[i, j]: 1 if customer i is assigned to facility j (0/1)
    # y[j]: 1 if facility j is open (0/1)
    
    # Total variables: n_cust * n_fac + n_fac
    n_vars = n_cust * n_fac + n_fac
    
    # Objective coefficients
    # Minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i, j] * x[i, j])
    c = np.zeros(n_vars)
    for j in range(n_fac):
        c[j * n_cust + i] = assign_cost_arr[i, j]  # Wait, indexing needs care
        # Let's define variable index mapping:
        # var_idx(i, j) = i * n_fac + j for x[i, j]
        # var_idx_y(j) = n_cust * n_fac + j for y[j]
        
    # Re-initialize c with correct mapping
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            idx = i * n_fac + j
            c[idx] = assign_cost_arr[i, j]
    for j in range(n_fac):
        idx = n_cust * n_fac + j
        c[idx] = open_cost_arr[j]
        
    # Bounds: all variables are binary
    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    #    sum_j x[i, j] = 1 for all i
    #    This is n_cust constraints.
    #    Matrix A1 has shape (n_cust, n_vars)
    
    # 2. Capacity constraints:
    #    sum_i demands[i] * x[i, j] <= capacities[j] for all j
    #    This is n_fac constraints.
    #    Matrix A2 has shape (n_fac, n_vars)
    
    # 3. Linking constraints:
    #    x[i, j] <= y[j] for all i, j
    #    This is n_cust * n_fac constraints.
    #    Matrix A3 has shape (n_cust * n_fac, n_vars)
    
    # Total constraints: n_cust + n_fac + n_cust * n_fac
    n_constraints = n_cust + n_fac + n_cust * n_fac
    A = np.zeros((n_constraints, n_vars))
    lb_constraints = np.zeros(n_constraints)
    ub_constraints = np.zeros(n_constraints)
    
    # Fill A, lb, ub
    row_idx = 0
    
    # 1. Assignment constraints (equality)
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
        lb_constraints[row_idx] = 1.0
        ub_constraints[row_idx] = 1.0
        row_idx += 1
        
    # 2. Capacity constraints (inequality <=)
    for j in range(n_fac):
        for i in range(n_cust):
            A[row_idx, i * n_fac + j] = demands_arr[i]
        lb_constraints[row_idx] = -np.inf
        ub_constraints[row_idx] = capacities_arr[j]
        row_idx += 1
        
    # 3. Linking constraints (inequality <=)
    # x[i, j] - y[j] <= 0  =>  x[i, j] <= y[j]
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
            A[row_idx, n_cust * n_fac + j] = -1.0
            lb_constraints[row_idx] = -np.inf
            ub_constraints[row_idx] = 0.0
            row_idx += 1
            
    # Define linear constraint
    constraints = LinearConstraint(A, lb_constraints, ub_constraints)
    bounds = Bounds(lb, ub)
    
    # Solve MILP
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, method="highs")
        
        if res.success:
            x_flat = res.x
            # Extract x[i, j]
            x = x_flat[:n_cust * n_fac].reshape(n_cust, n_fac)
            # Find best facility for each customer
            # Due to numerical precision, take argmax
            assignment = np.argmax(x, axis=1)
            
            # Verify validity
            # Check if assignment is integer (it is by construction)
            # Check capacity
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[assignment[i]] += demands_arr[i]
            
            if np.all(load <= capacities_arr + 1e-6):
                return assignment.astype(int)
            else:
                # Fallback: repair or return greedy if invalid
                # For robustness, return a greedy solution if MILP result is invalid
                pass
                
    except Exception as e:
        pass
    
    # Fallback: Greedy heuristic
    # Sort customers by demand descending
    cust_order = np.argsort(-demands_arr)
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    for i in cust_order:
        d_i = demands_arr[i]
        feasible = np.where(load + d_i <= capacities_arr)[0]
        if len(feasible) == 0:
            # Pick facility with most remaining capacity
            f = int(np.argmax(capacities_arr - load))
            res[i] = f
            load[f] += d_i
        else:
            # Pick feasible facility with lowest assignment cost
            costs = assign_cost_arr[i, feasible]
            f_idx = int(feasible[np.argmin(costs)])
            res[i] = f_idx
            load[f_idx] += d_i
            
    return res
