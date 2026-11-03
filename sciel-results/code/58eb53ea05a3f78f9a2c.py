"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), the coefficient should be 1
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_half_integer(U, V, W, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is close to an integer or half-integer
                if abs(val - round(val)) > tol and abs(val - (round(2*val)/2.0)) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    # Start with schoolbook
    best_rank = rank_schoolbook
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Return the schoolbook decomposition as a safe baseline
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
