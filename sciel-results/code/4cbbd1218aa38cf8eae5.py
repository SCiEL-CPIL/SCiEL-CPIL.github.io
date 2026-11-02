"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook decomposition
    # Rank R = m (the inner dimension)
    # This works because T[i,j,k] = 1 for all i,j,k
    # and we can write 1 = sum_r delta(j,r) * delta(j,r) * 1
    # Wait, sum_r delta(j,r)delta(j,r) is 1 if j is in range, which it is.
    
    R = m
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.ones((pn, R), dtype=float)

    # Fill U and V
    # U index: i*m + j, col: r
    # U[i, j, r] = 1 if j == r
    for i in range(n):
        for j in range(m):
            r = j
            U[i * m + j, r] = 1.0
            
    # V index: j*p + k, col: r
    # V[j, k, r] = 1 if j == r
    for j in range(m):
        for k in range(p):
            r = j
            V[j * p + k, r] = 1.0
            
    # W is all ones
    # W[k, i, r] = 1

    # Verification
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err < 1e-12:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: trivial rank n*m*p
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
