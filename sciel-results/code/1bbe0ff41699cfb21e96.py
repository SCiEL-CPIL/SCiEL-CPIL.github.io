"""Matrix multiplication 3x3x5 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    n, m, p = DIMENSIONS
    # The multiplication tensor T[(i',j'),(j'',k''),(i'',k'')] = delta[i',i''] * delta[k'',k'']
    # factors as (1_j' tensor I_n) kron (1_j'' tensor I_p), rank = n*p.
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
            W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
