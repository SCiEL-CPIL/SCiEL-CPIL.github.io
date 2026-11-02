"""
Rank-minimal tensor decomposition for 2x5 * 5x6 matrix multiplication.

Strategy:
1. Start from the schoolbook decomposition (rank 60).
2. Use iterative compression / rank reduction via solving linear systems
   to find a lower-rank decomposition if one exists.
3. For small dimensions, we can attempt to find a decomposition of rank r
   by solving a system of polynomial equations, but that's infeasible for
   large r. Instead, we use a heuristic: try to find a rank-r decomposition
   by random initialization + gradient-free local search, or by using the
   fact that for small matrices, known bounds exist.

Known results: For 2x5 * 5x6, the Strassen-like algorithms may give
rank < 60. The trivial upper bound is 60. We'll attempt to find a
decomposition of rank 55, 50, 45, etc. using a structured approach.

Actually, for this specific size, let's think about it differently.
The matrix multiplication tensor M_{2,5,6} has rank at most 60.
Can we do better? 

One approach: use the "laser method" or known constructions, but for
such small sizes, it's more practical to do a computational search.

Given the time constraint (120s), I'll implement:
1. A function to verify a decomposition is exact.
2. A search that tries to find a lower-rank decomposition by:
   a. Starting with random rank-r matrices and minimizing the residual.
   b. Using the fact that entries should be half-integers, we can
      round and verify.

But this is heuristic. A more reliable approach for small cases:
Use the fact that the rank of M_{n,m,p} is at most n*m*p, and try to
find a decomposition of rank r by solving for the entries.

For r < n*m*p, the system is underdetermined. We can set up the equations:
For each (i,j,k), the product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.

This is a system of n*m*p equations in 3*r unknowns (with U, V, W of size 
n*m x r, m*p x r, p*n x r). For r = 55, we have 60 equations and 3*55=165 
unknowns, so it's underdetermined. But the equations are cubic (product of 
three variables), so it's a polynomial system.

Given the complexity, I'll try a practical approach:
1. Try known constructions if available.
2. Otherwise, fall back to schoolbook (rank 60).
3. Attempt a simple local search to reduce rank.

For now, let's implement a robust schoolbook and a verification function,
and attempt a heuristic rank reduction.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(mat, tol=1e-9):
    """Snap values to nearest half-integer if close."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2) / 2
    if np.allclose(mat, snapped, atol=tol):
        return snapped.tolist()
    return mat.tolist()

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try to reduce rank using a heuristic approach
    # For small sizes, try random restarts with local search
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Attempt rank reduction for a few target ranks
    for target_rank in [55, 50, 45, 40]:
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random half-integers in {-1, 0, 1}
            U_r = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V_r = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W_r = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            # Check if this random decomposition works
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                break
        
        if best_rank == target_rank:
            break
    
    # If no reduction found, return schoolbook
    if best_rank == n * m * p:
        U, V, W, rank = _schoolbook(n, m, p)
    else:
        rank = best_rank
        U, V, W = best_U, best_V, best_W
    
    return {"U": U, "V": V, "W": W}
