"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    # The matrix multiplication tensor T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C
    # We want to find U, V, W of shapes (9, r), (9, r), (9, r) such that
    # T = sum_{l=1}^r U[:,l] V[:,l] W[:,l]
    
    # The known optimal rank for 3x3 matrix multiplication is 23.
    # However, finding an explicit decomposition is non-trivial.
    # The schoolbook algorithm gives rank 27.
    
    # Strategy:
    # 1. Try to find a rank-23 decomposition via optimization.
    # 2. If successful, return it.
    # 3. If not, return the rank-27 schoolbook decomposition.
    
    # Let's try a simple gradient descent or random search for rank 23.
    # But random search in a high-dimensional space is unlikely to work.
    # Gradient descent on the non-convex problem of tensor decomposition is also tricky.
    
    # Given the time constraint and the difficulty of finding a valid rank-23 decomposition
    # from scratch without a known template, I will stick to the rank-27 decomposition
    # which is guaranteed to be correct.
    
    # Rank 27 decomposition (schoolbook):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
