import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3x8.
    Uses the schoolbook decomposition (Rank 48) with robust NumPy output.
    """
    start_time = time.time()
    # Deadline guard: Evaluator kills at 120s. We stop at 100s to be safe.
    deadline = start_time + 100.0

    n, m, p = DIMENSIONS
    rank = n * m * p  # 48

    # Use integer arrays to satisfy "integer/half-integer entries" exactly
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Standard schoolbook term: e_i^T U e_j * e_j^T V e_k * e_k^T W e_i
                # Indices:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1

    # Check time; if we had a search algorithm, we would run it here.
    # Since we are using the known schoolbook decomposition, we return immediately.
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
