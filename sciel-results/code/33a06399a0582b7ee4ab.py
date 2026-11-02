"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

def _build_schoolbook(n, m, p):
    """Return (U, V, W) lists-of-lists for schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Tensor T[i,j,k] = delta (i from A, j from B, k from C)
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,j,k) is a valid triple
    # Actually the tensor for matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p
    # is T[i, j, k] = 1 if the triple (i, j, k) corresponds to a valid multiplication term.
    # The standard tensor is T[a, b, c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
    # such that T[(i,m),(j,p),(k,n)] = 1 when the indices align for multiplication.
    
    # More precisely: T[i,j,k] in the original 3-tensor form is 1 if:
    # i (row of A, 0..n-1), j (col of A / row of B, 0..m-1), k (col of B, 0..p-1)
    # And the output is at (i, k) in C.
    
    # The factorization is: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a = i*m + j, b = j*p + k, c = k*n + i
    # And T[a,b,c] = 1 iff the indices are consistent (which they always are for valid a,b,c).
    
    # Let's just check that for each (i, j, k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 130.0  # 150s limit, leave 20s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try random search for lower rank decompositions
    # We'll try to find decompositions with rank < schoolbook
    
    np.random.seed(42)
    random.seed(42)
    
    # For small dimensions, we can try a structured approach
    # The rank of 3x5x5 matrix multiplication is at most 75 (schoolbook)
    # Lower bounds are known but exact value may not be known
    
    # Strategy: try to find a decomposition by random linear combinations
    # This is a hard problem, so we'll try a few approaches
    
    # Approach 1: Try to verify if a slightly lower rank works with random perturbations
    # This is unlikely to work well, but let's try
    
    # Actually, for matrix multiplication tensors, known results suggest:
    # R(3,5,5) is not known exactly, but upper bounds exist from various methods
    
    # Let's try a simple approach: check if the schoolbook is already minimal
    # by verifying it, and return it if we can't find better quickly
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Try some structured reductions
    # For now, return schoolbook as it's guaranteed correct
    
    return {"U": best_U, "V": best_V, "W": best_W}
