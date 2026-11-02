"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use the standard Kronecker-product-style rank-n*m*p decomposition.
    # For C[i,k] = sum_j A[i,j] B[j,k] with tensor T[i,j,k]=1,
    # the trivial decomposition has one rank-1 term per (i,j,k) triple.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
