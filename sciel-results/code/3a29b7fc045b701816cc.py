import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 60  # Leave 30s buffer for 90s limit
    
    # Objective: maximize sum(1/i * x_i)
    # scipy.optimize.milp minimizes, so we minimize -sum(1/i * x_i)
    c = -np.array([1.0 / i for i in range(1, n + 1)])
    
    # Variables: x_1, ..., x_n, each in {0, 1}
    integrality = np.ones(n)
    lb = np.zeros(n)
    ub = np.ones(n)
    
    # Constraints: For each 3-AP (a, b, c), x_a + x_b + x_c <= 2
    # Indices in 0-based array: a-1, b-1, c-1
    # We can build the constraint matrix row by row or use sparse format.
    # Number of 3-APs is approx n^2/4. For n=256, ~16k constraints.
    # Dense matrix of size 16k x 256 is 16k * 256 * 8 bytes ~ 32MB. This is fine.
    
    ap_rows = []
    for a in range(1, n + 1):
        for d in range(1, n - a + 1):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            # Constraint: x_{a-1} + x_{b-1} + x_{c-1} <= 2
            row = np.zeros(n)
            row[a - 1] = 1
            row[b - 1] = 1
            row[c - 1] = 1
            ap_rows.append(row)
            
    if not ap_rows:
        # No constraints? Unlikely for n>2
        return list(range(1, n + 1))
        
    A = np.array(ap_rows)
    b_ub = np.full(len(ap_rows), 2)
    
    # Linear constraint: A @ x <= b_ub
    # In scipy.optimize.milp, constraints are defined as:
    # linear_constraint: LinearConstraint(A, lb, ub)
    # We want A @ x <= 2, so lb = -inf, ub = 2
    constraints = LinearConstraint(A, -np.inf, b_ub)
    
    # Bounds for variables
    var_bounds = Bounds(lb, ub)
    
    try:
        # Solve ILP
        # time_limit is in seconds
        res = milp(c=c,
                   constraints=constraints,
                   integrality=integrality,
                   bounds=var_bounds,
                   options={"time_limit": 55, "mip_rel_gap": 1e-6})
        
        if res.status == 0: # Optimal
            x = res.x
            # Round to nearest integer to be safe, though they should be 0 or 1
            x_int = np.round(x).astype(int)
            
            # Verify validity
            selected = [i + 1 for i in range(n) if x_int[i] == 1]
            if not selected:
                # Fallback if empty
                selected = [1]
            else:
                # Check for 3-APs
                s = set(selected)
                valid = True
                for a in s:
                    for b in s:
                        if a >= b: continue
                        c = 2 * b - a
                        if c in s and c != a and c != b:
                            valid = False
                            break
                    if not valid: break
                
                if not valid:
                    # If solver returned invalid solution (shouldn't happen with exact ILP, but floating point issues?)
                    # Fallback to greedy
                    selected = []
                    for i in range(1, n + 1):
                        if all(2*b-a != i for a in selected for b in selected if a < b):
                            selected.append(i)
                            
            return selected
            
        elif res.status == 1: # Feasible but not optimal (time limit reached)
            # res.x might contain a feasible integer solution if the solver found one
            if res.x is not None:
                x_int = np.round(res.x).astype(int)
                selected = [i + 1 for i in range(n) if x_int[i] == 1]
                
                # Verify
                s = set(selected)
                valid = True
                for a in s:
                    for b in s:
                        if a >= b: continue
                        c = 2 * b - a
                        if c in s and c != a and c != b:
                            valid = False
                            break
                    if not valid: break
                
                if valid and selected:
                    return selected
            
            # Fallback to greedy if no good solution
            selected = []
            for i in range(1, n + 1):
                if all(2*b-a != i for a in selected for b in selected if a < b):
                    selected.append(i)
            return selected
        else:
            # Error or infeasible
            pass
            
    except Exception as e:
        pass

    # Fallback: Greedy
    selected = []
    for i in range(1, n + 1):
        if all(2*b-a != i for a in selected for b in selected if a < b):
            selected.append(i)
    return selected
