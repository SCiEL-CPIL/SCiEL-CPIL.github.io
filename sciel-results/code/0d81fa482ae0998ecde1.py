import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Trivial schoolbook decomposition (rank 75)
    def trivial_decomposition():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W, rank

    # Verify decomposition
    def verify_decomposition(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Return trivial decomposition
    U, V, W, rank = trivial_decomposition()
    
    if not verify_decomposition(U, V, W, rank):
        raise ValueError("Decomposition verification failed")

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
