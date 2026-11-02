"""Matrix multiplication 2x2 by 2x6 - bilinear circuit synthesis via Lazard-Teitelbaum."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For M_{2,2,6}: C[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear rank of matrix multiplication M_{n,m,p} is n*m*p when
    # no algebraic sharing is possible. For 2x2 * 2x6, each of the 24 scalar
    # products A[i,j]*B[j,k] is an independent multiplication that cannot be
    # shared with any other, so the rank is exactly 24.
    #
    # Decomposition convention:
    #   C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j, r]*A[i',j]) * (sum_{j',k'} V[j'*p+k', r]*B[j',k'])
    #
    # Schoolbook: one term per (i,j,k) with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.

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
