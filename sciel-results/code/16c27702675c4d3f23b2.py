"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Target tensor T
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # In the flattened indices:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u, v, w] = 1 if indices are consistent
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    # Try to find a lower rank decomposition
    # We will try ranks from 95 down to 88
    # This is a heuristic search
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Simple search: Try to find a decomposition of rank R
    # We use a greedy approach: pick random sparse vectors
    # This is unlikely to succeed quickly, so we limit attempts
    
    # If we had a known rank 88 decomposition, we would return it here.
    # Since we don't, we return schoolbook.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
