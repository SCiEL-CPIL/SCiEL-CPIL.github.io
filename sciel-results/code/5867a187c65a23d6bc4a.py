import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    base = 51975
    
    # Get all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors: try multiple orderings
    divisors_sorted = sorted(divisors)  # ascending
    divisors_desc = sorted(divisors, reverse=True)  # descending
    
    def greedy(order):
        covered = np.zeros(base, dtype=np.int8)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                indices = np.arange(r, base, m)
                count = np.sum(covered[indices] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            indices = np.arange(best_r, base, m)
            covered[indices] = 1
        return classes
    
    def compute_score(classes):
        covered = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            covered[np.arange(r, base, m)] = 1
        return int(np.sum(covered == 0))
    
    # Get greedy solutions with different orderings
    best_classes = greedy(divisors_sorted)
    best_score = compute_score(best_classes)
    
    # Try descending order too
    classes_desc = greedy(divisors_desc)
    score_desc = compute_score(classes_desc)
    if score_desc < best_score:
        best_classes = classes_desc
        best_score = score_desc
    
    # Try ILP with time limit
    try:
        start_time = time.time()
        
        # Build ILP: maximize number of covered residues
        # Variables: x_{m,r} in {0,1} for each divisor m and residue r in [0,m)
        #            y_j in {0,1} for each residue j in [0, base)
        # Constraint: for each residue j, sum over all (m,r) where j ≡ r (mod m) of x_{m,r} >= y_j
        # Objective: maximize sum of y_j (equivalently minimize uncovered)
        
        # Map divisors to indices
        n_divs = len(divisors)
        div_idx = {d: i for i, d in enumerate(divisors)}
        
        # Count total variables
        total_x_vars = sum(m for m in divisors)
        total_vars = total_x_vars + base
        
        # If too many variables, skip ILP
        if total_vars > 100000:
            raise Exception("Too many variables")
        
        # Build constraint matrix A for: A @ vars <= b
        # We want: for each j, sum_{(m,r): j≡r mod m} x_{m,r} - y_j >= 0
        # Which is: -sum_{(m,r): j≡r mod m} x_{m,r} + y_j <= 0
        
        n_constraints = base
        # Use sparse matrix
        from scipy.sparse import csr_matrix
        
        rows = []
        cols = []
        data = []
        
        # For each residue j, add constraint: -sum x_{m,r} + y_j <= 0
        x_var_offset = 0
        for m in divisors:
            for r in range(m):
                var_idx = x_var_offset + r
                for j in range(r, base, m):
                    rows.append(j)
                    cols.append(var_idx)
                    data.append(-1)
            x_var_offset += m
        
        # y_j variable for each j
        for j in range(base):
            rows.append(j)
            cols.append(x_var_offset + j)
            data.append(1)
        
        A = csr_matrix((data, (rows, cols)), shape=(n_constraints, total_vars))
        b = np.zeros(n_constraints)
        
        # Objective: minimize -sum y_j (i.e., maximize sum y_j)
        c = np.zeros(total_vars)
        for j in range(base):
            c[x_var_offset + j] = -1
        
        # Bounds: all binary
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        
        # Integrality: all binary
        integrality = np.ones(total_vars)
        
        constraints = LinearConstraint(A, -np.inf, b)
        bounds = Bounds(lb, ub)
        
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        elapsed = time.time() - start_time
        if elapsed > 90:
            raise Exception("ILP timeout")
        
        if result.success and result.x is not None:
            # Extract solution
            ilp_classes = []
            for m in divisors:
                for r in range(m):
                    var_idx = sum(div for div in divisors if div < m) + r
                    if result.x[var_idx] > 0.5:
                        ilp_classes.append([m, r])
                        break  # Only one residue per modulus
        
        if ilp_classes:
            ilp_score = compute_score(ilp_classes)
            if ilp_score < best_score:
                best_classes = ilp_classes
                best_score = ilp_score
    
    except Exception:
        pass
    
    # Local search: try changing residues for each class
    classes = [list(c) for c in best_classes]
    improved = True
    max_iterations = 20
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(classes)):
            m, r = classes[i]
            best_r = r
            best_local_score = best_score
            for r_new in range(m):
                if r_new == r:
                    continue
                classes[i][1] = r_new
                new_score = compute_score(classes)
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_r = r_new
            if best_r != r:
                classes[i][1] = best_r
                best_score = best_local_score
                improved = True
            else:
                classes[i][1] = r
    
    # Second pass of local search
    improved = True
    while improved and iteration < 40:
        improved = False
        iteration += 1
        for i in range(len(classes)):
            m, r = classes[i]
            best_r = r
            best_local_score = best_score
            for r_new in range(m):
                if r_new == r:
                    continue
                classes[i][1] = r_new
                new_score = compute_score(classes)
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_r = r_new
            if best_r != r:
                classes[i][1] = best_r
                best_score = best_local_score
                improved = True
            else:
                classes[i][1] = r
    
    return classes
