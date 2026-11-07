import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Variables: x_ij for i in 0..n_customers-1, j in 0..n_facilities-1
    # y_j for j in 0..n_facilities-1
    # Total vars: n_customers * n_facilities + n_facilities
    
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    c = np.zeros(n_vars)
    # Objective: sum assign_cost[i][j] * x_ij + sum open_cost[j] * y_j
    for i in range(n_customers):
        for j in range(n_facilities):
            c[i * n_facilities + j] = assign_cost[i][j]
    for j in range(n_facilities):
        c[n_x + j] = open_cost[j]
    
    # Constraints:
    # 1. Sum_j x_ij = 1 for each i
    # 2. Sum_i demands[i] * x_ij <= capacities[j] for each j
    # 3. x_ij - y_j <= 0 for each i, j
    
    A_list = []
    lb_list = []
    ub_list = []
    
    # Constraint 1: Assignment
    for i in range(n_customers):
        row = np.zeros(n_vars)
        for j in range(n_facilities):
            row[i * n_facilities + j] = 1.0
        A_list.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)
        
    # Constraint 2: Capacity
    for j in range(n_facilities):
        row = np.zeros(n_vars)
        for i in range(n_customers):
            row[i * n_facilities + j] = demands[i]
        A_list.append(row)
        lb_list.append(0.0)
        ub_list.append(capacities[j])
        
    # Constraint 3: Linking x and y
    for i in range(n_customers):
        for j in range(n_facilities):
            row = np.zeros(n_vars)
            row[i * n_facilities + j] = 1.0
            row[n_x + j] = -1.0
            A_list.append(row)
            lb_list.append(-np.inf)
            ub_list.append(0.0)
            
    A = np.array(A_list)
    lb = np.array(lb_list)
    ub = np.array(ub_list)
    
    constraints = LinearConstraint(A, lb, ub)
    integrality = np.ones(n_vars)
    bounds = Bounds(0, 1)
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0: # Success
            x_flat = res.x[:n_x]
            # x_flat is 0 or 1 (or close to it)
            assignment = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                for j in range(n_facilities):
                    if x_flat[i * n_facilities + j] > 0.5:
                        assignment[i] = j
                        break
            return assignment
        else:
            # Fallback if MIP fails
            pass
    except Exception:
        pass
        
    # Fallback: Greedy
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result
