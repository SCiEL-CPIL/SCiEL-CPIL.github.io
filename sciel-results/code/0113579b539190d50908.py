"""Matrix multiplication 3x5x5 with a rank-75 schoolbook decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-75 decomposition for 3x5 * 5x5 matrix multiplication.

    The multiplication tensor T has entries indexed by (i, j, k) with
    i in [0, n), j in [0, m), k in [0, p).  The bilinear form is

        C[i, k] = sum_{j} A[i, j] * B[j, k]

    and the tensor satisfies
        T[i, j, k] = 1   for every (i, j, k)
    in the sense that each rank-1 term picks out exactly one (i, j, k)
    triple.  The schoolbook algorithm uses one rank term per (i, j, k),
    giving rank = n * m * p = 75.

    Factor matrices (as required by the interface):
        U : (n*m) x rank   -> U[i*m + j, c]
        V : (m*p) x rank   -> V[j*p + k, c]
        W : (p*n) x rank   -> W[k*n + i, c]

    For each rank term c corresponding to (i, j, k):
        U[i*m + j, c] = 1
        V[j*p + k, c] = 1
        W[k*n + i, c] = 1
    and all other entries in that column are 0.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
