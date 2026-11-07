"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Fallback greedy solution (same as seed, but robust)
    def greedy_fallback():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # If no facility has capacity, assign to one with most remaining capacity 
                # or just the cheapest (this would be invalid, but we assume feasible instance)
                # To ensure validity, we might need to relax or pick best fit
                # The problem statement implies a valid assignment exists.
                # If we fail here, we return an invalid result, which scores 0.
                # We should try to find *any* valid assignment.
                # Let's try to find the facility with max remaining capacity.
                rem_cap = capacities - load
                f = np.argmax(rem_cap)
                if rem_cap[f] >= demands[i]:
                    load[f] += demands[i]
                    result[i] = f
                else:
                    # Should not happen if instance is feasible
                    result[i] = 0 
        return result

    start_time = time.time()
    timeout = 10.0 # Safety margin
    
    try:
        if hasattr(scipy.optimize, 'milp'):
            # Variables: x[i, j] for i in range(n_customers), j in range(n_facilities)
            #           y[j] for j in range(n_facilities)
            # Total variables: N = n_customers * n_facilities + n_facilities
            
            N = n_customers * n_facilities + n_facilities
            
            # Objective coefficients
            c = np.zeros(N)
            for i in range(n_customers):
                for j in range(n_facilities):
                    idx = i * n_facilities + j
                    c[idx] = assign_cost[i, j]
            for j in range(n_facilities):
                idx = n_customers * n_facilities + j
                c[idx] = open_cost[j]
            
            # Integrality: all integer
            integrality = np.ones(N, dtype=int)
            
            # Bounds: all [0, 1]
            bounds = [(0, 1)] * N
            
            # Constraints
            
            # 1. Equality: Each customer assigned to exactly one facility
            #    sum_j x[i, j] = 1
            A_eq = np.zeros((n_customers, N))
            b_eq = np.ones(n_customers)
            for i in range(n_customers):
                for j in range(n_facilities):
                    idx = i * n_facilities + j
                    A_eq[i, idx] = 1.0
            
            # 2. Inequality: Linking x[i, j] <= y[j]  =>  x[i, j] - y[j] <= 0
            #    and Capacity: sum_i demands[i] * x[i, j] <= capacities[j]
            
            n_link_constraints = n_customers * n_facilities
            n_cap_constraints = n_facilities
            total_ineq = n_link_constraints + n_cap_constraints
            
            A_ineq = np.zeros((total_ineq, N))
            b_ineq = np.zeros(total_ineq)
            
            # Linking constraints
            row = 0
            for i in range(n_customers):
                for j in range(n_facilities):
                    idx_x = i * n_facilities + j
                    idx_y = n_customers * n_facilities + j
                    A_ineq[row, idx_x] = 1.0
                    A_ineq[row, idx_y] = -1.0
                    b_ineq[row] = 0.0
                    row += 1
            
            # Capacity constraints
            for j in range(n_facilities):
                for i in range(n_customers):
                    idx_x = i * n_facilities + j
                    A_ineq[row, idx_x] = demands[i]
                b_ineq[row] = capacities[j]
                row += 1
            
            constraints = [
                scipy.optimize.LinearConstraint(A_eq, lb=1.0, ub=1.0),
                scipy.optimize.LinearConstraint(A_ineq, lb=-np.inf, ub=b_ineq)
            ]
            
            # Solve
            res = scipy.optimize.milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"presolve": True})
            
            if res.success:
                x_sol = res.x
                # Extract x matrix
                x_mat = x_sol[:n_customers * n_facilities].reshape(n_customers, n_facilities)
                # Assign to facility with x > 0.5 (should be 1)
                assignment = np.argmax(x_mat, axis=1)
                
                # Verify validity
                load = np.zeros(n_facilities)
                for i in range(n_customers):
                    f = assignment[i]
                    load[f] += demands[i]
                if np.all(load <= capacities):
                    return assignment.astype(int)
            
            # Check timeout
            if time.time() - start_time > timeout:
                return greedy_fallback()
                
        else:
            # Fallback if milp not available
            return greedy_fallback()
            
    except Exception:
        pass
        
    # Final fallback
    if time.time() - start_time > timeout:
         return greedy_fallback()
         
    return greedy_fallback()

# EVOLVE-BLOCK-END
