import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    n = 256
    # Weights: 1/x for x in 1..n
    weights = [1.0 / x for x in range(1, n + 1)]
    
    # Generate all 3-APs
    # A 3-AP is (a, b, c) such that a + c = 2b
    # We store indices (0-based) for the constraint matrix
    ap_indices = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Convert to 0-based indices
            ap_indices.append((a - 1, b - 1, c - 1))
            
    num_constraints = len(ap_indices)
    
    # Prepare data for scipy.optimize.milp
    # Objective: minimize -sum(weights * x)
    c = np.array([-w for w in weights])
    
    # Constraint matrix A
    # Each row corresponds to an AP (a, b, c)
    # Constraint: x_a + x_b + x_c <= 2
    # We use LinearConstraint(A, lb, ub)
    # A is (num_constraints, n)
    # We can build it as a sparse matrix for efficiency
    
    rows = []
    cols = []
    data = []
    
    for i, (a, b, c_idx) in enumerate(ap_indices):
        rows.extend([i, i, i])
        cols.extend([a, b, c_idx])
        data.extend([1, 1, 1])
        
    from scipy.sparse import csr_matrix
    A = csr_matrix((data, (rows, cols)), shape=(num_constraints, n))
    
    # Bounds for constraints: 0 <= x_a + x_b + x_c <= 2
    # Actually, x_i >= 0 is handled by variable bounds.
    # The constraint is x_a + x_b + x_c <= 2.
    # LinearConstraint(A, lb, ub)
    # lb = 0 (since x_i >= 0, sum >= 0)
    # ub = 2
    constraints = LinearConstraint(A, lb=0, ub=2)
    
    # Variable bounds: 0 <= x_i <= 1
    var_bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer (0 or 1)
    integrality = np.ones(n)
    
    best_solution = None
    best_score = 0.0
    
    try:
        # Try exact ILP solve
        # time_limit in seconds
        options = {'time_limit': 80.0}
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=var_bounds, options=options)
        
        if res.status == 0: # Optimal
            sol = res.x
            # Convert to set
            S = set()
            for i in range(n):
                if sol[i] > 0.5:
                    S.add(i + 1)
            
            # Validate
            if is_valid_3ap_free(S, n):
                score = sum(1.0 / x for x in S)
                if score > best_score:
                    best_score = score
                    best_solution = S
        elif res.status == 1: # Feasible but not optimal (time limit reached)
            sol = res.x
            S = set()
            for i in range(n):
                if sol[i] > 0.5:
                    S.add(i + 1)
            if is_valid_3ap_free(S, n):
                score = sum(1.0 / x for x in S)
                if score > best_score:
                    best_score = score
                    best_solution = S
    except Exception as e:
        # Fallback to heuristic if ILP fails
        pass
    
    if best_solution is not None and len(best_solution) > 0:
        return sorted(best_solution)
        
    # Fallback Heuristic: Greedy + Local Search from current program
    if time.time() > deadline - 5:
        # Return a simple valid set if time is almost up
        return [1]
        
    # Greedy
    greedy_set = set()
    for x in range(1, n + 1):
        if can_add_fast(x, greedy_set, n):
            greedy_set.add(x)
            
    best_set = greedy_set.copy()
    best_weight = sum(1.0 / x for x in best_set)
    
    # Local Search
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 10 and iteration < 50:
        improved = False
        iteration += 1
        
        # Phase 1: Add
        for x in range(1, n + 1):
            if x not in best_set and can_add_fast(x, best_set, n):
                best_set.add(x)
                best_weight += 1.0 / x
                improved = True
                
        # Phase 2: Swap
        # Try to replace larger numbers with smaller ones if possible
        # This is a simplified version of the current program's swap logic
        for x in range(1, n + 1):
            if x in best_set:
                continue
            # Try to find a y in best_set such that removing y allows adding x
            # And 1/x > 1/y (i.e., x < y)
            for y in range(x + 1, n + 1):
                if y in best_set:
                    temp_set = best_set - {y}
                    if can_add_fast(x, temp_set, n):
                        best_set = temp_set
                        best_set.add(x)
                        best_weight += (1.0 / x - 1.0 / y)
                        improved = True
                        break
            if improved:
                break
                
    return sorted(best_set)

def can_add_fast(x, S, n):
    """
    Check if x can be added to S without forming a 3-AP.
    S is a set of ints.
    """
    # Check if x is middle: a + c = 2x
    for a in S:
        c = 2 * x - a
        if c in S and c != a:
            return False
            
    # Check if x is endpoint
    # x, b, c => 2b = x + c => b = (x+c)/2
    # a, b, x => 2b = a + x => b = (a+x)/2
    for a in S:
        s = x + a
        if s % 2 == 0:
            b = s // 2
            if b in S and b != x and b != a:
                return False
                
    return True

def is_valid_3ap_free(S, n):
    """
    Verify that S contains no 3-AP.
    """
    S_list = sorted(S)
    S_set = set(S_list)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            # Check if c = 2b - a is in S and c > b
            c = 2 * b - a
            if c > b and c in S_set:
                return False
    return True
