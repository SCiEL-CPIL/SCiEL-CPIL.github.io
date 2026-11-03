"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # Target: find rank-minimal decomposition
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # We'll search for a sparse solution with half-integer entries
    # Start with a known good upper bound and try to improve
    
    # For 4x5x5, try rank 20 first as a starting point
    # Use a greedy structured approach
    
    n, m, p = 4, 5, 5
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    # The tensor T has entries T[(i,j), (j,k), (k,i)] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # and = 0 otherwise (but actually the tensor is only defined for valid indices)
    
    # Actually, the matrix multiplication tensor has T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try to find a rank-20 solution
    # Strategy: for each r, pick a pattern of (i,j,k) triples that form a "line" or structured set
    
    # Try rank = 20: one term per (i,k) pair? No, that doesn't work directly.
    
    # Alternative: use the fact that we can have U[i*m+j, r] depend on i and j,
    # V[j*p+k, r] depend on j and k, W[k*n+i, r] depend on k and i.
    
    # Let's try a structured ansatz:
    # For r = 0, 1, ..., R-1, define:
    # U[i*m+j, r] = a_r(i,j)
    # V[j*p+k, r] = b_r(j,k)  
    # W[k*n+i, r] = c_r(k,i)
    # such that sum_r a_r(i,j) b_r(j,k) c_r(k,i) = 1
    
    # A simple construction: use 5 terms per "diagonal" of the j-k plane
    # For each j, we need to handle 5 k's. 
    
    # Let's try: rank = n * m = 20, with one term per (i,j)
    # U[i*m+j, r] = 1 if r == i*m+j else 0
    # V[j*p+k, r] = 1 for all k (when r corresponds to this j)
    # W[k*n+i, r] = 1 if this (i,k) is valid
    # But then sum_r gives sum over r, and for fixed (i,j,k), only r=i*m+j contributes,
    # giving 1*1*1 = 1. This works! Rank = 20.
    
    # Wait, let me verify: for fixed (i,j,k), the sum is over r.
    # U[i*m+j, r] is nonzero only when r = i*m+j, where it's 1.
    # V[j*p+k, r] when r = i*m+j: this depends on j and k. If we set V[j*p+k, i*m+j] = 1,
    # then V[j*p+k, r] = 1 when r encodes the same j.
    # W[k*n+i, r] when r = i*m+j: set to 1.
    
    # So: U[i*m+j, i*m+j] = 1, all other U entries 0
    # V[j*p+k, i*m+j] = 1 for all i,k (for fixed j)
    # W[k*n+i, i*m+j] = 1 for all i,k (for fixed j)
    
    # Check: for (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # = U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # = 1 * 1 * 1 = 1. Correct!
    
    # But wait, V[j*p+k, i*m+j] must be well-defined. For a given j, there are n=4 values of i,
    # so there are 4 different r values that have the same j. V[j*p+k, r] should be 1 for all
    # r that correspond to this j, regardless of i.
    
    # Similarly for W.
    
    # This gives rank = n*m = 20.
    
    rank = n * m  # 20
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
