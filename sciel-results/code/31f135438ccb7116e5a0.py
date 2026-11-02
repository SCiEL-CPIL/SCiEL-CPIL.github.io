"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s/150s wall-clock limits

    n, m, p = DIMENSIONS
    rank = n * m * p  # 216, the schoolbook rank for 6x6x6

    # Build the canonical schoolbook decomposition:
    # For each (i, j, k) triple, one rank term with:
    #   U[i*m + j, col] = 1
    #   V[j*p + k, col] = 1
    #   W[k*n + i, col] = 1
    # This exactly reproduces the matrix multiplication tensor under the
    # trace convention W[k*n + i, r].
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Deadline guard: construction is O(rank) deterministic, finishes in microseconds.
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
