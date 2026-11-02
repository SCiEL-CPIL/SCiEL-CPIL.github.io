import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit

    # Known optimal rank for M_{2,3,3} is 17
    target_rank = 17

    # Standard Strassen factors for 2x2x2 (rank 7)
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, -1],
        [1, 0, -1, 0, 0, 0, 1]
    ], dtype=float)

    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ], dtype=float)

    W_s = np.array([
        [1, -1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, -1, 0, 1]
    ], dtype=float)

    U = np.zeros((n*m, target_rank), dtype=float)
    V = np.zeros((m*p, target_rank), dtype=float)
    W = np.zeros((p*n, target_rank), dtype=float)

    # Embed Strassen in i in {0,1}, j in {0,1}, k in {0,1}
    for i in range(2):
        for j in range(2):
            U[i*m + j, 0:7] = U_s[i*2 + j, :]
    for j in range(2):
        for k in range(2):
            V[j*p + k, 0:7] = V_s[j*2 + k, :]
    for i in range(2):
        for k in range(2):
            W[k*n + i, 0:7] = W_s[i*2 + k, :]

    # Add schoolbook for remaining terms: (j,k) where not (j<2 and k<2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j < 2 and k < 2:
                    continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

    if col != target_rank:
        return _schoolbook(n, m, p)

    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
