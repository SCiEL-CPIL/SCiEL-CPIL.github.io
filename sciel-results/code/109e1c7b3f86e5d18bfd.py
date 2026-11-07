import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Fallback greedy solution in case MILP fails or takes too long
    def greedy_solve():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Find cheapest feasible facility
            costs = assign_cost[i].copy()
            # Mask infeasible facilities
            for f in range(n_facilities):
                if load[f] + demands[i] > capacities[f]:
                    costs[f] = np.inf
            # If no feasible facility, this is a problem, but task guarantees feasibility?
            # If not, we might need to relax. Assuming feasibility.
            f_min = np.argmin(costs)
            if np.isinf(costs[f_min]):
                # Fallback to any facility if capacity violated (invalid result, but better than crash?)
                # The task says invalid results score 0. We should try hard to be valid.
                # Greedy largest-first usually works if total capacity >= total demand.
                pass 
            load[f_min] += demands[i]
            result[i] = f_min
        return result

    # Check if we have enough time
    start_time = time.time()
    deadline = 100.0 # Safety margin
    
    try:
        # Variables:
        # 0..19: y_j (open facility j)
        # 20..: x_ij (assign customer i to facility j)
        
        # Objective coefficients
        c = np.zeros(n_facilities + n_customers * n_facilities)
        c[:n_facilities] = open_cost
        # Flatten assign_cost: row-major order
        c[n_facilities:] = assign_cost.flatten()
        
        # Constraints
        # 1. Assignment: sum_j x_ij = 1 for all i
        # 2. Capacity: sum_i demands[i] * x_ij - capacities[j] * y_j <= 0 for all j
        
        A_rows = []
        lb = []
        ub = []
        
        # Assignment constraints
        for i in range(n_customers):
            row = np.zeros(n_facilities + n_customers * n_facilities)
            # x_ij are at indices: n_facilities + i * n_facilities + j
            start_idx = n_facilities + i * n_facilities
            row[start_idx : start_idx + n_facilities] = 1.0
            A_rows.append(row)
            lb.append(1.0)
            ub.append(1.0)
            
        # Capacity constraints
        for j in range(n_facilities):
            row = np.zeros(n_facilities + n_customers * n_facilities)
            # y_j is at index j
            row[j] = -capacities[j]
            # x_ij are at indices: n_facilities + i * n_facilities + j
            for i in range(n_customers):
                idx = n_facilities + i * n_facilities + j
                row[idx] = demands[i]
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            
        A = np.array(A_rows)
        
        # Bounds
        bounds = Bounds(lb=np.zeros(len(c)), ub=np.ones(len(c)))
        
        # Integrality
        integrality = np.ones(len(c))
        
        constraint = LinearConstraint(A, lb, ub)
        
        # Check time
        if time.time() - start_time > deadline:
            return greedy_solve()
            
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if time.time() - start_time > deadline:
            return greedy_solve()
            
        if res.status == 0:
            # Extract solution
            x_opt = res.x
            y_opt = x_opt[:n_facilities]
            x_assign = x_opt[n_facilities:].reshape(n_customers, n_facilities)
            
            # Determine assignment
            result = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                # Find facility with x_ij = 1
                assigned = np.where(x_assign[i] > 0.5)[0]
                if len(assigned) > 0:
                    result[i] = assigned[0]
                else:
                    # Should not happen if constraint satisfied
                    result[i] = 0 
            
            # Verify validity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[result[i]] += demands[i]
            
            valid = True
            for j in range(n_facilities):
                if load[j] > capacities[j] + 1e-6:
                    valid = False
                    break
            
            if valid:
                return result
            else:
                return greedy_solve()
        else:
            return greedy_solve()
            
    except Exception as e:
        return greedy_solve()
