import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time
from itertools import product

def construct():
    base = 51975
    
    # Get all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Precompute for each divisor m, the residues j in [0, base) for each r
    # This is needed for the ILP constraints
    
    # ILP formulation:
    # x_{m,r} binary: select residue r for modulus m
    # y_j binary: residue j is covered
    # Maximize sum y_j
    # s.t. sum_r x_{m,r} <= 1 for each m
    #      y_j <= sum_{(m,r): j ≡ r mod m} x_{m,r} for each j
    #      y_j >= 0
    
    # Variable ordering: first all x_{m,r}, then all y_j
    x_vars = []
    for m in divisors:
        for r in range(m):
            x_vars.append((m, r))
    n_x = len(x_vars)
    
    y_vars = list(range(base))
    n_y = len(y_vars)
    
    n_vars = n_x + n_y
    
    # Objective: maximize sum y_j => minimize -sum y_j
    c = np.zeros(n_vars)
    c[n_x:] = -1.0  # minimize -sum(y_j) = maximize sum(y_j)
    
    # Constraints:
    # 1. For each m: sum_r x_{m,r} <= 1
    # 2. For each j: y_j - sum_{(m,r): j≡r mod m} x_{m,r} <= 0
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    # Constraint 1: sum_r x_{m,r} <= 1 for each m
    idx = 0
    for m in divisors:
        row = np.zeros(n_vars)
        for r in range(m):
            row[idx] = 1.0
            idx += 1
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(1.0)
    
    # For each residue j, build constraint: y_j - sum x_{m,r} <= 0
    # i.e., y_j <= sum x_{m,r} where j ≡ r mod m
    # We need to map (m, r) to its index in x_vars
    x_index = {}
    idx = 0
    for m in divisors:
        for r in range(m):
            x_index[(m, r)] = idx
            idx += 1
    
    for j in range(base):
        row = np.zeros(n_vars)
        row[n_x + j] = 1.0  # y_j
        for m in divisors:
            r = j % m
            row[x_index[(m, r)]] = -1.0
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(0.0)
    
    A = np.array(A_rows)
    lb = np.array(b_lower)
    ub = np.array(b_upper)
    
    # Bounds: all variables in [0, 1]
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Integrality: x vars and y vars are binary
    integrality = np.ones(n_vars)
    
    constraints = LinearConstraint(A, lb, ub)
    
    start_time = time.time()
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 100, "mip_rel_gap": 0.01}
        )
        
        elapsed = time.time() - start_time
        
        if result.status == 0 or result.status == 1:  # optimal or best found
            x_vals = result.x[:n_x]
            classes = []
            idx = 0
            for m in divisors:
                best_r = -1
                best_val = -1
                for r in range(m):
                    if x_vals[idx] > 0.5:
                        best_r = r
                        break
                    idx += 1
                if best_r >= 0:
                    classes.append([m, best_r])
            
            # Compute score
            covered = np.zeros(base, dtype=np.int8)
            for m, r in classes:
                covered[np.arange(r, base, m)] = 1
            score = int(np.sum(covered == 0))
            
            return classes
        else:
            raise Exception(f"ILP failed with status {result.status}")
    
    except Exception as e:
        # Fallback: greedy approach
        pass
    
    # Greedy fallback
    covered = np.zeros(base, dtype=np.int8)
    classes = []
    remaining = list(divisors)
    
    while remaining:
        best_m = None
        best_r = None
        best_count = -1
        for m in remaining:
            for r in range(m):
                indices = np.arange(r, base, m)
                count = np.sum(covered[indices] == 0)
                if count > best_count:
                    best_count = count
                    best_m = m
                    best_r = r
        if best_m is None:
            break
        classes.append([best_m, best_r])
        indices = np.arange(best_r, base, best_m)
        covered[indices] = 1
        remaining.remove(best_m)
    
    # Local search improvement
    def compute_score(cls):
        cov = np.zeros(base, dtype=np.int8)
        for m, r in cls:
            cov[np.arange(r, base, m)] = 1
        return int(np.sum(cov == 0))
    
    best_score = compute_score(classes)
    improved = True
    max_iter = 5
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(len(classes)):
            m, r = classes[i]
            best_r = r
            best_local = best_score
            for r_new in range(m):
                if r_new == r:
                    continue
                classes[i][1] = r_new
                s = compute_score(classes)
                if s < best_local:
                    best_local = s
                    best_r = r_new
            if best_r != r:
                classes[i][1] = best_r
                best_score = best_local
                improved = True
            else:
                classes[i][1] = r
    
    return classes
