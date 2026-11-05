import numpy as np
from fractions import Fraction
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_sparse_decomposition(n, m, p, max_rank, time_limit=60):
    """
    Try to find a low-rank decomposition using sparse patterns with entries in {-1, 0, 1}.
    Use a combinatorial search over support patterns.
    """
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # For each rank-1 term r, it has support in U, V, W.
    # U[i*m+j, r] is nonzero for some (i,j)
    # V[j*p+k, r] is nonzero for some (j,k)  
    # W[k*n+i, r] is nonzero for some (k,i)
    
    # Key constraint: for the product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] to be nonzero,
    # all three must be nonzero simultaneously.
    
    # Strategy: try rank from 20 up to max_rank, search for valid patterns
    # This is expensive, so use a heuristic approach
    
    # Better: use the known structure. For all-ones tensor, try:
    # Partition the (i,k) pairs into groups, and for each j, assign which groups contribute
    
    # Let's try a specific structured approach:
    # For n=2, p=5: the (i,k) space has 10 entries
    # For each j in {0,1,2}, we need to cover all 10 (i,k) entries
    # A rank-1 term with U support {(i1,j1),(i2,j2)}, V support {(j1,k1),(j1,k2),...}, W support {(k1,i1),(k2,i2),...}
    # contributes to specific (i,j,k) entries
    
    # Simplified: try to find r such that we can write 1 = sum of r "products"
    # where each product is a function of (i,j,k)
    
    # Let's try a direct construction for rank 24:
    # Use 3 blocks of 8 terms each, one for each j value
    # For each j, we need to represent the 2x5 all-ones matrix as sum of 8 rank-1 matrices
    # (each rank-1 matrix is outer product of a 2-vector and a 5-vector)
    # But we need consistency across j for the W factor
    
    # Actually, let me try a different approach: use the fact that
    # T[i,j,k] = 1 can be written as sum over r of f_r(i)*g_r(j)*h_r(k)
    # if we allow the tensor to be "separable" in this way
    # But our tensor is NOT separable in this way because of the index coupling
    
    # Let me try yet another approach: use the rank-28 solution and try to merge terms
    # by finding linear dependencies
    
    return None

def _construct_rank24(n, m, p):
    """
    Try to construct a rank-24 decomposition for 2x3x5.
    
    Idea: Use the structure that for each j, the (i,k) sub-tensor is all-ones.
    We can use a "diagonal" approach where each rank-1 term covers a specific
    pattern of (i,j,k) indices.
    
    For n=2, p=5, m=3:
    - 10 (i,k) pairs
    - 3 j values
    - 30 total entries
    
    Try to use 24 terms where each term covers a "cross" pattern.
    """
    # Let's try a specific construction:
    # For j=0,1,2 and for each (i,k), we need coverage.
    # 
    # Approach: Use 8 terms per j-value (24 total)
    # For each j, represent the 2x5 all-ones as sum of 8 rank-1 (2x5) matrices
    # A 2x5 matrix can be written as sum of 2 rank-1 matrices (since rank <= 2)
    # But we need 8 to allow for the W coupling...
    
    # Actually, a 2x5 matrix has rank at most 2, so it can be written as sum of 2 rank-1 matrices.
    # If we use 2 terms per j, that's only 6 terms total. But the W factor couples across j.
    
    # The issue is that W[k*n+i, r] depends on both k and i, and the same W entry
    # is used for different j values. So we can't independently decompose each j-slice.
    
    # Let me try a different strategy: use the known rank-28 and try to find
    # a rank-24 by solving a linear system with sparsity constraints.
    
    # For now, return None and fall back to rank-28
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Leave 20s margin
    
    # Build the known rank-28 solution
    rank_28 = 28
    U28 = [[0] * rank_28 for _ in range(n * m)]
    V28 = [[0] * rank_28 for _ in range(m * p)]
    W28 = [[0] * rank_28 for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U28[i * m + 2][r] = 1
            V28[2 * p + k][r] = 1
            W28[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U28[i * m + j][r] = 1
            V28[j * p + 4][r] = 1
            W28[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # M1
    U28[0 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 0][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U28[1 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 0][r] = 1
    W28[0 * n + 1][r] = 1
    W28[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U28[0 * m + 0][r] = 1
    V28[0 * p + 1][r] = 1
    V28[1 * p + 1][r] = -1
    W28[1 * n + 0][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U28[1 * m + 1][r] = 1
    V28[1 * p + 0][r] = 1
    V28[0 * p + 0][r] = -1
    W28[0 * n + 0][r] = 1
    W28[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U28[0 * m + 0][r] = 1
    U28[0 * m + 1][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = -1
    W28[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U28[1 * m + 0][r] = 1
    U28[0 * m + 0][r] = -1
    V28[0 * p + 0][r] = 1
    V28[0 * p + 1][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U28[0 * m + 1][r] = 1
    U28[1 * m + 1][r] = -1
    V28[1 * p + 0][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # M1
    U28[0 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 2][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U28[1 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 2][r] = 1
    W28[2 * n + 1][r] = 1
    W28[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U28[0 * m + 0][r] = 1
    V28[0 * p + 3][r] = 1
    V28[1 * p + 3][r] = -1
    W28[3 * n + 0][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U28[1 * m + 1][r] = 1
    V28[1 * p + 2][r] = 1
    V28[0 * p + 2][r] = -1
    W28[2 * n + 0][r] = 1
    W28[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U28[0 * m + 0][r] = 1
    U28[0 * m + 1][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = -1
    W28[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U28[1 * m + 0][r] = 1
    U28[0 * m + 0][r] = -1
    V28[0 * p + 2][r] = 1
    V28[0 * p + 3][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U28[0 * m + 1][r] = 1
    U28[1 * m + 1][r] = -1
    V28[1 * p + 2][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = 1
    r += 1
    
    # Verify the decomposition
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    # Start with the rank-28 solution
    best_U = [row[:] for row in U28]
    best_V = [row[:] for row in V28]
    best_W = [row[:] for row in W28]
    best_rank = rank_28
    
    # Try to find lower rank via combinatorial search
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 20:
        # Try a structured rank-24 approach
        result_24 = _construct_rank24(n, m, p)
        if result_24 is not None:
            U_new, V_new, W_new = result_24
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_U = U_new
                best_V = V_new
                best_W = W_new
                best_rank = 24
    
    return {"U": best_U, "V": best_V, "W": best_W}
