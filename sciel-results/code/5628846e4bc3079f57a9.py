"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    """
    Greedy heuristic: Assign largest demands first to the cheapest feasible facility.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Sort facilities by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # If no facility found, assign to the one with most remaining capacity 
            # (though this might violate capacity, it's better than crashing)
            # In a valid problem, this shouldn't happen if total capacity >= total demand
            # We pick the facility with max remaining capacity as a safety net
            rem = capacities - load
            f = np.argmax(rem)
            load[f] += demands[i]
            result[i] = f
            
    return result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Try MIP
    try:
        # Variables: x_{ij} for i in n_cust, j in n_fac (n_cust * n_fac vars)
        #             y_j for j in n_fac (n_fac vars)
        # Total vars: n_cust * n_fac + n_fac
        
        # Objective coefficients
        c = np.zeros(n_cust * n_fac + n_fac)
        
        # x_{ij} coefficients are assign_cost[i, j]
        # y_j coefficients are open_cost[j]
        
        # Flatten assign_cost: row-major order (i, j)
        c[:n_cust * n_fac] = assign_cost.flatten()
        c[n_cust * n_fac:] = open_cost
        
        # Constraints
        # 1. Assignment: sum_j x_{ij} = 1 for all i
        #    Matrix A_eq shape: (n_cust, n_cust*n_fac + n_fac)
        A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
        for i in range(n_cust):
            # x_{ij} are at indices i*n_fac + j
            for j in range(n_fac):
                A_eq[i, i * n_fac + j] = 1.0
        
        b_eq = np.ones(n_cust)
        
        # 2. Capacity: sum_i demands[i] * x_{ij} <= capacities[j] * y_j for all j
        #    sum_i demands[i] * x_{ij} - capacities[j] * y_j <= 0
        #    Matrix A_ub shape: (n_fac, n_cust*n_fac + n_fac)
        A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
        for j in range(n_fac):
            # Coefficients for x_{ij}
            for i in range(n_cust):
                A_ub[j, i * n_fac + j] = demands[i]
            # Coefficient for y_j
            # y_j is at index n_cust*n_fac + j
            A_ub[j, n_cust * n_fac + j] = -capacities[j]
            
        b_ub = np.zeros(n_fac)
        
        # Bounds
        # x_{ij} in [0, 1]
        # y_j in [0, 1]
        integrality = np.zeros(n_cust * n_fac + n_fac)
        integrality[:] = 1  # All binary
        
        lb = np.zeros(n_cust * n_fac + n_fac)
        ub = np.ones(n_cust * n_fac + n_fac)
        
        # Define constraints
        constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)
        constraint_ub = LinearConstraint(A_ub, -np.inf, b_ub)
        
        bounds = Bounds(lb, ub)
        
        # Solve
        res = milp(
            c=c,
            constraints=[constraint_eq, constraint_ub],
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            x = res.x
            # Extract x_{ij}
            x_mat = x[:n_cust * n_fac].reshape(n_cust, n_fac)
            # Find assigned facility for each customer
            # Since it's binary, argmax should work if solution is clean
            # Sometimes MIP returns 0.9999 or similar
            result = np.argmax(x_mat, axis=1).astype(int)
            
            # Verify validity
            load = np.zeros(n_fac)
            for i, f in enumerate(result):
                load[f] += demands[i]
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                # Fallback if MIP returned invalid due to numerical issues
                return _greedy_fallback(demands, capacities, open_cost, assign_cost)
        else:
            # MIP failed, fallback
            return _greedy_fallback(demands, capacities, open_cost, assign_cost)
            
    except Exception:
        # If MIP throws an error, fallback
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)

# EVOLVE-BLOCK-END
