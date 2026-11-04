"""Exact tensor decomposition for 2x3 times 3x5 matrix multiplication.

Implements the rank-30 Kronecker-product construction: each term is an outer product
of elementary row-selection, column-selection, and output-index vectors.
Uses NumPy for clean construction, with a deadline guard for safety.
"""

import time
import numpy as np

def construct():
    # Deadline guard: return before the 150 s wall-clock limit.
    deadline = time.time() + 120.0

    n, m, p = 2, 3, 5
    rank = n * m * p  # 30

    # Factor matrices: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U indexes input row i, column j of A (shape n x m)
                U[i * m + j, col] = 1
                # V indexes input row j, column k of B (shape m x p)
                V[j * p + k, col] = 1
                # W indexes output: trace convention W[k*n + i, r]
                W[k * n + i, col] = 1
                col += 1

        if time.time() > deadline:
            break

    # Convert to list-of-lists of Python ints for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
