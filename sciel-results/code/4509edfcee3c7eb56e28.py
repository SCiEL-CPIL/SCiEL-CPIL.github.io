"""Direct construction of the exact tensor decomposition for (2,3,5) matrix multiplication.

Strategy: rank-30 trivial decomposition using Kronecker-product structure of identity
matrices. Each term corresponds to an elementary row-selection (U), column-selection (V),
and output-index (W) vector with entries 0 or 1. This gives an exact integer decomposition
with rank = n*m*p = 30, which matches the known upper bound R(2,3,5) <= 2*R(1,3,5) = 30.

The construction is O(n*m*p) in time and memory, completing in well under 1 ms.
A deadline guard is included as a safety margin per the compute budget directive.
"""

import time

def construct():
    # Deadline guard: ensure we return before the 150 s wall-clock limit.
    deadline = time.time() + 120.0  # 120 s safety margin (evaluator kills at 150 s)

    n, m, p = 2, 3, 5
    rank = n * m * p  # 30

    # Factor matrices: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U indexes input row i, column j of A (shape n x m)
                U[i * m + j][col] = 1
                # V indexes input row j, column k of B (shape m x p)
                V[j * p + k][col] = 1
                # W indexes output: trace convention W[k*n + i, r]
                W[k * n + i][col] = 1
                col += 1

        # Deadline check (redundant for this O(1) construction, but per directive)
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
