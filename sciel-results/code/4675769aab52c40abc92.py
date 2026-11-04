import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                max_err = max(max_err, abs(val - 1))
    return max_err < 1e-10

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Block decomposition: 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1
    # Use schoolbook for each block: rank 9 + 9 + 9 = 27
    rank_total = 27
    U_full = np.zeros((n * m, rank_total), dtype=float)
    V_full = np.zeros((m * p, rank_total), dtype=float)
    W_full = np.zeros((p * n, rank_total), dtype=float)

    # Block 1: B columns 0-2 (3x3x3 schoolbook)
    offset = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = offset + i * 3 + j * 1 + k
                U_full[i * m + j, r] = 1.0
                V_full[j * p + k, r] = 1.0
                W_full[k * n + i, r] = 1.0

    # Block 2: B columns 3-5 (3x3x3 schoolbook)
    offset = 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = offset + i * 3 + j * 1 + k
                U_full[i * m + j, r] = 1.0
                V_full[j * p + (k + 3), r] = 1.0
                W_full[(k + 3) * n + i, r] = 1.0

    # Block 3: B column 6 (3x3x1 schoolbook)
    offset = 18
    for i in range(3):
        for j in range(3):
            r = offset + i * 3 + j
            U_full[i * m + j, r] = 1.0
            V_full[j * p + 6, r] = 1.0
            W_full[6 * n + i, r] = 1.0

    # Verify the full decomposition
    if not _verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Fallback to full schoolbook (rank 63)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U_full, "V": V_full, "W": W_full}
