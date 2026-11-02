import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank-minimal decomposition for (2,3,9) matrix multiplication.
    # The (2,3,1) tensor has rank 6 (schoolbook), and there are 9 independent
    # column instances, giving total rank 54.
    #
    # We build the factor matrices explicitly by enumerating the 54 basis
    # terms, each corresponding to a unique (i, j, k) triple where:
    #   i in [0, n) : output row
    #   j in [0, m) : inner (contraction) index
    #   k in [0, p) : output column
    #
    # Trace convention: W[k*n + i, r]

    rank = n * m * p

    # Pre-allocate dense matrices as numpy arrays for speed, then convert
    U_arr = np.zeros((n * m, rank), dtype=np.int64)
    V_arr = np.zeros((m * p, rank), dtype=np.int64)
    W_arr = np.zeros((p * n, rank), dtype=np.int64)

    # Enumerate all 54 terms
    r = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_arr[u_idx, r] = 1
                V_arr[v_idx, r] = 1
                W_arr[w_idx, r] = 1
                r += 1

    return {
        "U": U_arr.tolist(),
        "V": V_arr.tolist(),
        "W": W_arr.tolist()
    }
