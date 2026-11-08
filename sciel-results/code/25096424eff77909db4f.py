"""Rank-48 exact tensor decomposition for 2x5 @ 5x6 via block Strassen."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_add(U, V, W, i0, i1, j0, j1, k0, k1, col):
    """Add 7 Strassen products for A[i0:i1+1, j0:j1+1] @ B[j0:j1+1, k0:k1+1]
    into C[i0:i1+1, k0:k1+1].  U rows indexed i*m+j, V rows j*p+k, W rows k*n+i.
    Returns next free column."""
    # P1 = (A_i0j0 + A_i1j1) * (B_j0k0 + B_j1k1)
    U[i0*m + j0][col] += 1; U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1; V[j1*p + k1][col] += 1
    W[k0*n + i0][col] += 1; W[k1*n + i1][col] += 1
    col += 1
    # P2 = (A_i1j0 + A_i1j1) * B_j0k0
    U[i1*m + j0][col] += 1; U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + i1][col] += 1; W[k1*n + i1][col] -= 1
    col += 1
    # P3 = A_i0j0 * (B_j0k1 - B_j1k1)
    U[i0*m + j0][col] += 1
    V[j0*p + k1][col] += 1; V[j1*p + k1][col] -= 1
    W[k1*n + i0][col] += 1
    col += 1
    # P4 = A_i1j1 * (B_j1k0 - B_j0k0)
    U[i1*m + j1][col] += 1
    V[j1*p + k0][col] += 1; V[j0*p + k0][col] -= 1
    W[k0*n + i1][col] += 1
    col += 1
    # P5 = (A_i0j0 + A_i0j1) * B_j1k1
    U[i0*m + j0][col] += 1; U[i0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] -= 1; W[k1*n + i0][col] += 1
    col += 1
    # P6 = (A_i1j0 - A_i0j0) * (B_j0k0 + B_j0k1)
    U[i1*m + j0][col] += 1; U[i0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1; V[j0*p + k1][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    # P7 = (A_i0j1 - A_i1j1) * (B_j1k0 + B_j1k1)
    U[i0*m + j1][col] += 1; U[i1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1; V[j1*p + k1][col] += 1
    W[k0*n + i0][col] += 1
    col += 1
    return col


def construct():
    n, m, p = DIMENSIONS
    # Partition intermediate dimension (size 5) into j-blocks and output (size 6) into k-blocks
    j_blocks = [(0, 1), (2, 3), (4, 4)]   # sizes 2, 2, 1
    k_blocks = [(0, 1), (2, 3), (4, 5)]   # sizes 2, 2, 2

    rank = 0
    for (j0, j1) in j_blocks:
        for (k0, k1) in k_blocks:
            if j1 == j0:
                rank += 2 * (k1 - k0 + 1)
            else:
                rank += 7 * (k1 - k0 + 1)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (j0, j1) in j_blocks:
        for (k0, k1) in k_blocks:
            if j1 == j0:
                # Single intermediate column j0: C[i,k] = A[i,j0]*B[j0,k]
                for i in range(n):
                    for k in range(k0, k1 + 1):
                        U[i * m + j0][col] += 1
                        V[j0 * p + k][col] += 1
                        W[k * n + i][col] += 1
                        col += 1
            else:
                # 2x2 Strassen block (i in {0,1}, j in {j0,j1}, k in {k0,k1})
                col = _strassen_add(U, V, W, 0, 1, j0, j1, k0, k1, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
