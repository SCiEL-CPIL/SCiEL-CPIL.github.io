import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square (diagonal constraint a=a => 2a not square)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1] # Fallback, though 1 is valid (2!=sq)
        
    # Map vertex to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build constraint matrix for MILP
    # We want to maximize sum(x_i)
    # Constraints:
    # 1. x_i + x_j <= 1 for all pairs (i, j) such that valid_vertices[i] + valid_vertices[j] is a square.
    #    Note: This includes i=j? No, diagonal is already filtered out.
    #    If i != j and v_i + v_j is square, they cannot both be in the set.
    #    So x_i + x_j <= 1.
    
    # We can build the A matrix for scipy.milp.
    # scipy.milp takes A_ub, b_ub for A_ub @ x <= b_ub.
    # Each constraint is a row in A_ub.
    
    # To avoid building a sparse matrix manually which might be slow or error-prone with dense numpy for 200 vars,
    # we can just list the constraints.
    # Number of constraints is roughly O(n^2). 200^2 = 40000. This is small.
    
    rows = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                # Constraint: x_i + x_j <= 1
                # Create a row of zeros, put 1 at i and 1 at j
                # Instead of creating full dense rows in a list (memory heavy if many), 
                # we can use scipy.sparse if available, but standard numpy is fine for 200x40000? 
                # 40000 * 200 * 8 bytes = 64 MB. Acceptable.
                # But constructing 40k dense rows of length 200 is slow in python loops.
                
                # Let's try to be smarter.
                # Actually, we can just collect the pairs and build the matrix.
                pass
    
    # Alternative: Use a dense matrix and fill it.
    # A_ub will be (num_constraints, n_vars)
    # We don't know num_constraints exactly until we iterate.
    # Let's estimate or just use a list of tuples and convert.
    
    constraint_pairs = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                constraint_pairs.append((i, j))
                
    m_constraints = len(constraint_pairs)
    
    if m_constraints == 0:
        # No conflicts between distinct vertices. All valid vertices can be selected.
        return valid_vertices
        
    # Initialize A_ub
    A_ub = np.zeros((m_constraints, n_vars), dtype=np.float64)
    b_ub = np.ones(m_constraints, dtype=np.float64)
    
    for k, (i, j) in enumerate(constraint_pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    # Objective: Maximize sum(x) => Minimize -sum(x)
    c = -np.ones(n_vars, dtype=np.float64)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer (0 or 1)
    integrality = np.ones(n_vars, dtype=np.int8)
    
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A_ub, ub=b_ub)],
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 40} # Leave 15s for overhead and return
        )
        
        if res.status == 0 or res.status == 1: # Optimal or best found within limit
            # res.x contains the solution
            # Filter out numerical noise
            solution = []
            for i in range(n_vars):
                if res.x[i] > 0.5:
                    solution.append(valid_vertices[i])
            
            if solution:
                return solution
            else:
                # Fallback if solver returned empty or failed
                return [valid_vertices[0]]
        else:
            # Solver failed or timed out without a solution? 
            # Fallback to a simple greedy to ensure non-empty valid set
            S = set()
            for v in valid_vertices:
                can_add = True
                for u in S:
                    if u + v in squares:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            return list(S)
            
    except Exception:
        # Fallback to greedy
        S = set()
        for v in valid_vertices:
            can_add = True
            for u in S:
                if u + v in squares:
                    can_add = False
                    break
            if can_add:
                S.add(v)
        return list(S)
