import time
import random
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 256

def _is_3ap_free(s):
    """
    Verify that the set s contains no nonconstant 3-term arithmetic progression.
    """
    if not s:
        return True
    s_set = set(s)
    lst = list(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            a, b = lst[i], lst[j]
            # Check if a, b, c is AP with c > b
            c = 2 * b - a
            if c > b and c in s_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin before 90s limit
    
    # 1. Build Conflict Graph
    # We have N nodes (1 to N).
    # An edge exists between u and v if they form a 3-AP with some w in [1, N].
    # Specifically, if there exists w such that {u, v, w} is a 3-AP.
    # This is equivalent to checking if u, v are in the same 3-AP.
    # For ILP, we need constraints: x_u + x_v <= 1 for every conflicting pair (u, v).
    
    # Efficiently find all conflicting pairs.
    # A 3-AP is defined by (a, d) where a, a+d, a+2d are in [1, N].
    # The pairs are (a, a+d), (a, a+2d), (a+d, a+2d).
    # We can just iterate through all possible 3-APs and add edges.
    
    # Using a set of tuples (min(u,v), max(u,v)) to store edges
    edges = set()
    
    # Iterate over all possible 3-APs
    # a + 2d <= N
    # d >= 1
    # a >= 1
    
    max_d = (N - 1) // 2
    for d in range(1, max_d + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            
            # Pairs: (a,b), (a,c), (b,c)
            edges.add((a, b))
            edges.add((a, c))
            edges.add((b, c))
            
    # Convert to lists for matrix construction
    # We need a matrix A where A[i, u] + A[i, v] <= 1 for each edge (u, v).
    # Actually, standard form for MILP: A @ x <= b.
    # For each edge (u, v), we have a row with 1 at u-1 and 1 at v-1, and b=1.
    
    # Number of constraints = len(edges)
    # Number of variables = N
    m = len(edges)
    n_vars = N
    
    # We can use scipy.sparse for efficiency, but milp accepts dense too.
    # With N=256 and m ~ 3 * (sum of d) ~ 3 * 256^2/4 ~ 50k constraints.
    # A dense 50k x 256 matrix is 12.8M floats ~ 100MB. This is acceptable.
    # However, sparse is better.
    
    import scipy.sparse as sp
    
    # Initialize sparse matrix
    # Rows: 0 to m-1
    # Cols: 0 to N-1
    row_idx = []
    col_idx = []
    data = []
    
    for i, (u, v) in enumerate(edges):
        # u, v are 1-based
        # Convert to 0-based
        u0 = u - 1
        v0 = v - 1
        row_idx.extend([i, i])
        col_idx.extend([u0, v0])
        data.extend([1, 1])
        
    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(m, n_vars))
    
    # Objective: Maximize sum(1/x * x_i)
    # c = [1/1, 1/2, ..., 1/N]
    c = np.array([1.0 / i for i in range(1, N + 1)])
    
    # Constraints:
    # 1. A @ x <= 1 (for each edge)
    # 2. 0 <= x_i <= 1 (bounds)
    # Since it's a binary problem, we set integrality to 1 for all variables.
    
    constraints = LinearConstraint(A, ub=1)
    
    # Bounds: x_i in [0, 1]
    res_bounds = Bounds(lb=0, ub=1)
    
    # Integrality: 1 means binary
    integrality = np.ones(n_vars)
    
    try:
        # Solve MILP
        # HiGHS is the default solver for milp in scipy
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=res_bounds
        )
        
        if result.success:
            # Extract solution
            # result.x is the vector of x_i
            sol = result.x
            # Threshold to binary
            selected = [i + 1 for i in range(N) if sol[i] > 0.5]
            
            # Validate
            if _is_3ap_free(selected):
                # Return the set
                # The task asks for a nonempty subset.
                if not selected:
                    return [1]
                return selected
            else:
                # Fallback if validation fails (should not happen)
                pass
                
    except Exception as e:
        print(f"ILP failed: {e}")
        
    # Fallback: Greedy heuristic if ILP fails or times out
    # We have limited time left, so use a fast greedy
    s = []
    s_set = set()
    # Greedy ascending (prioritizes small numbers for high weight)
    for x in range(1, N + 1):
        can_add = True
        for y in s:
            z1 = 2 * x - y
            if z1 in s_set:
                can_add = False
                break
            if (x + y) % 2 == 0:
                z2 = (x + y) // 2
                if z2 in s_set and z2 != x and z2 != y:
                    can_add = False
                    break
        if can_add:
            s.append(x)
            s_set.add(x)
            
    if not s:
        return [1]
        
    return s

if __name__ == "__main__":
    res = construct()
    print(res)
    # Print score
    score = sum(1.0/x for x in res)
    print(f"Score: {score}")
    print(f"Size: {len(res)}")
    print(f"Valid: {_is_3ap_free(res)}")
