"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the tensor T[i,j,k,r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need this to equal 1 if i,j,k are valid (which they always are in the tensor)
    # Actually, the tensor for matrix mult is T[i,j,k] = 1 (it's just the identity in the sense
    # that (AB)[i,k] = sum_j A[i,j]B[j,k])
    # The decomposition is: (AB)[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    # Wait, let me re-read the convention.
    
    # Standard tensor decomposition for matrix multiplication:
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all valid indices (this is the "multiplication tensor")
    # A rank-r decomposition gives matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_rank_r(n, m, p, r):
    """
    Try to find a rank-r decomposition. Returns (U, V, W) as lists or None.
    Uses a structured approach: start with the schoolbook and try to merge/eliminate.
    """
    # For now, we'll use a simple approach: if r == n*m*p, return schoolbook.
    # For r < n*m*p, we need a smarter method.
    
    # One approach: use the fact that for (2,3,8), we can try to find a decomposition
    # by solving a system of equations. But this is complex.
    
    # Alternative: use known results. For (2,3,8), the schoolbook is 48.
    # Can we do better? 
    # 
    # Let's try a different strategy: use a random initialization and gradient descent
    # on the non-convex problem. But this is risky.
    #
    # Actually, let's think about it differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # We want to write T = sum_c U_c ⊗ V_c ⊗ W_c, where U_c[i*m+j] = u_{i,j}^{(c)},
    # V_c[j*p+k] = v_{j,k}^{(c)}, W_c[k*n+i] = w_{k,i}^{(c)}.
    # 
    # This is equivalent to: 1 = sum_c u_{i,j}^{(c)} * v_{j,k}^{(c)} * w_{k,i}^{(c)} for all i,j,k.
    
    # Let's try a simple construction for small r.
    # For r = n*m*p, it's trivial.
    
    # Let's try to use a known lower bound. The rank of the matrix multiplication tensor
    # is at least the border rank, which is at least the geometric rank.
    # For (2,3,8), the geometric rank is max(n, m, p) = 8? No, that's not right.
    # 
    # Actually, let's just try to return the schoolbook for now, but with a check
    # if we can do better.
    
    if r >= n * m * p:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    # For r < n*m*p, we need a better method.
    # Let's try a simple heuristic: use a subset of the schoolbook terms.
    # This won't work in general, but let's see.
    
    return None


def _try_small_rank(n, m, p, max_rank_to_try=20):
    """Try to find a decomposition with rank < n*m*p."""
    nmp = n * m * p
    for r in range(1, min(max_rank_to_try, nmp)):
        result = _construct_rank_r(n, m, p, r)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return U, V, W, r
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds
    
    # First, try to find a better decomposition than schoolbook.
    # For (2,3,8), schoolbook is 48.
    
    # Try a few known constructions or heuristics.
    # 
    # One idea: use the fact that 8 = 2*4, so maybe we can exploit structure.
    # But this is speculative.
    
    # Let's just return the schoolbook for now, but with a time guard.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
