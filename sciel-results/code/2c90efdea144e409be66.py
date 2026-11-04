import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin before 120s limit
    
    # We construct the standard schoolbook decomposition.
    # The tensor T has indices (i, j, k) with i in [0, n), j in [0, m), k in [0, p).
    # T[i, j, k] is the coefficient for A[i, j] * B[j, k] in C[i, k].
    # The decomposition is:
    # U[i*m + j, r] = 1 if r maps to (i, j, k), else 0
    # V[j*p + k, r] = 1 if r maps to (i, j, k), else 0
    # W[k*n + i, r] = 1 if r maps to (i, j, k), else 0
    # Note the W index convention: W[k*n + i, r]
    
    rank = n * m * p
    
    # Use integers for U, V, W entries to satisfy "integer/half-integer" strictly
    # and avoid any potential float comparison issues in the evaluator.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Mapping: column index r corresponds to tuple (i, j, k)
    # r = i * m * p + j * p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                # U index: i * m + j
                U[i * m + j][r] = 1
                # V index: j * p + k
                V[j * p + k][r] = 1
                # W index: k * n + i
                W[k * n + i][r] = 1
                
                # Check deadline, though this loop is O(18) and instantaneous
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
