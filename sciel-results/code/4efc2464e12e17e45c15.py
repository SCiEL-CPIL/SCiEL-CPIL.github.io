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
    
    def time_exceeded():
        return time.time() - start_time > timeout

    # Fallback heuristic if MIP fails
    def heuristic_fallback():
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        order = np.argsort(-demands)
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
        return res

    # Try MIP with improved parameter settings
    try:
        # Variables: x_{ij} (N*M), y_j (M)
        n_vars = n_customers * n_facilities + n_facilities
        
        # Objective: min sum(c_ij * x_ij) + sum(open_cost_j * y_j)
        c = np.zeros(n_vars)
        c[:n_customers * n_facilities] = assign_cost.flatten()
        c[n_customers * n_facilities:] = open_cost
        
        # Integrality: all binary
        integrality = np.ones(n_vars)
        
        # Bounds: [0, 1]
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        # Constraints
        # 1. Assignment: sum_j x_ij = 1 for each i
        # 2. Capacity: sum_i demand_i * x_ij <= capacity_j * y_j for each j
        
        A = np.zeros((n_customers + n_facilities, n_vars))
        lb = np.zeros(n_customers + n_facilities)
        ub = np.zeros(n_customers + n_facilities)
        
        # Assignment constraints
        for i in range(n_customers):
            for j in range(n_facilities):
                A[i, i * n_facilities + j] = 1.0
            lb[i] = 1.0
            ub[i] = 1.0
            
        # Capacity constraints
        for j in range(n_facilities):
            for i in range(n_customers):
                A[n_customers + j, i * n_facilities + j] = demands[i]
            A[n_customers + j, n_customers * n_facilities + j] = -capacities[j]
            lb[n_customers + j] = -np.inf
            ub[n_customers + j] = 0.0
            
        constraints = LinearConstraint(A, lb, ub)
        
        # Solve with time limit and improved options
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={'time_limit': timeout - (time.time() - start_time)})
        
        if res.success:
            x_flat = res.x[:n_customers * n_facilities]
            # Determine assignment
            assignment = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                # Find the facility with x_ij closest to 1
                x_i = x_flat[i * n_facilities : (i + 1) * n_facilities]
                assignment[i] = np.argmax(x_i)
            
            # Validate solution
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignment[i]] += demands[i]
            if np.all(load <= capacities + 1e-6):
                return assignment
            else:
                # If MIP solution is slightly infeasible due to numerical issues, try to fix or fallback
                # For now, fallback if invalid
                pass
                
    except Exception as e:
        pass

    # Fallback to heuristic
    return heuristic_fallback()
# EVOLVE-BLOCK-END
