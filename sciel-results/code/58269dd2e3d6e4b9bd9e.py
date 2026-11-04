"""Matrix multiplication 2x3x8 exact tensor decomposition (schoolbook, rank 48)."""

DIMENSIONS = [2, 3, 8]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS          # 2, 3, 8
    rank = n * m * p              # 48

    U = [[0] * rank for _ in range(n * m)]   # 6  x 48
    V = [[0] * rank for _ in range(m * p)]   # 24 x 48
    W = [[0] * rank for _ in range(p * n)]   # 16 x 48

    # Schoolbook: one rank-1 term per (i, j, k) triple.
    # U index: i*m + j ; V index: j*p + k ; W index: k*n + i (trace convention).
    idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
