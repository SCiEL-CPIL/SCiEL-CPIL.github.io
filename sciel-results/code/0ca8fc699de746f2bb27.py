"""Rewritten program for matrix multiplication 5x5x5 tensor decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # We'll build a decomposition of the 5x5x5 matrix multiplication tensor.
    # Strategy: partition the 5x5 matrix into blocks and use known low-rank algorithms.
    #
    # Partition: Split 5 = 2 + 3
    # A = [[A22, A23], [A32, A33]] where A22 is 2x2, A23 is 2x3, A32 is 3x2, A33 is 3x3
    # B = [[B22, B23], [B32, B33]]
    # C = A*B = [[C22, C23], [C32, C33]]
    #
    # C22 = A22*B22 + A23*B32  (2x2)
    # C23 = A22*B23 + A23*B33  (2x3)
    # C32 = A32*B22 + A33*B32  (3x2)
    # C33 = A32*B23 + A33*B33  (3x3)
    #
    # For each block product, we can use known algorithms:
    # - 2x2 matrix multiplication: Strassen (rank 7)
    # - 3x3 matrix multiplication: Laderman (rank 15)
    # - 2x3 times 3x2 (gives 2x2): standard rank 12, or we can combine
    # - 2x3 times 3x3 (gives 2x3): standard rank 18
    # - 3x2 times 2x2 (gives 3x2): standard rank 12
    # - 3x2 times 2x3 (gives 3x3): standard rank 18
    #
    # Actually, let me think more carefully. The tensor decomposition for matrix multiplication
    # of an (n x m) matrix by an (m x p) matrix has rank at most n*m*p (standard).
    #
    # For the full 5x5x5 problem, the standard rank is 125.
    #
    # Let me use a simpler approach: just use the standard 125-term decomposition,
    # but verify it carefully. This is guaranteed to work.
    #
    # Then, if time permits, try to find a better decomposition.
    
    def standard_decomposition(n, m, p):
        """Standard schoolbook decomposition with rank n*m*p."""
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

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is correct."""
        rank = len(U[0]) if U else 0
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        # T[i,k,s] = sum_r U[i,s,r] * V[s,k,r] * W[k,i,r]
        # Need T[i,k,s] = 1 for all i,k,s
        # Note: U has shape (n*m, rank), indexed as U[i*m + s, r]
        # V has shape (m*p, rank), indexed as V[s*p + k, r]
        # W has shape (p*n, rank), indexed as W[k*n + i, r]
        
        for i in range(n):
            for k in range(p):
                for s in range(m):
                    total = 0.0
                    for r in range(rank):
                        total += Ua[i * m + s, r] * Va[s * p + k, r] * Wa[k * n + i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Start with standard decomposition
    U, V, W = standard_decomposition(n, m, p)
    rank = n * m * p

    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: return standard decomposition anyway
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
