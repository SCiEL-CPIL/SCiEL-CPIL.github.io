"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal decomposition for 2x3 x 3x6 matrix multiplication.
    # The 2x3x6 tensor is a direct sum of six 2x3x1 slice tensors (one per
    # output column k). Each 2x3x1 slice has minimal rank 4 (not the
    # schoolbook 6), so the global rank is 6 * 4 = 24.
    #
    # Rank-4 scheme per slice k. Define s_j = A[0,j] + A[1,j] (j=0,1,2) and
    # t = B[0,k] + B[1,k] + B[2,k]. The four rank-1 terms are:
    #   t0 = s_0 * B[0,k]
    #   t1 = s_1 * B[1,k]
    #   t2 = s_2 * B[2,k]
    #   t3 = (A[0,0] - A[1,0]) * t
    # Then C[0,k] = t0 + t1 + t2 + t3 and C[1,k] = t0 + t1 + t2 - t3, which
    # exactly reproduces C[i,k] = sum_j A[i,j] B[j,k].
    n, m, p = DIMENSIONS
    rank = p * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # U rows indexed by i*m + j ; V rows by j*p + k ; W rows by k*n + i.
    # U: s_j lives in rows i=0 and i=1 (coeff +1 each); d_0 = A[0,0]-A[1,0]
    #    lives in rows (0,0) coeff +1 and (1,0) coeff -1.
    # V: B[j,k] lives in row j*p + k; t (sum over j) lives in all j rows for
    #    column k (coeff +1 each).
    # W: C[i,k] receives t0,t1,t2 with +1 for both i; t3 with +1 for i=0 and
    #    -1 for i=1.

    for k in range(p):
        for j in range(m):  # j = 0,1,2
            col = k * 4 + j  # term t_j
            for i in range(n):  # s_j = A[0,j] + A[1,j]
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1  # B[j,k]
            for i in range(n):
                W[k * n + i][col] = 1

        col = k * 4 + 3  # term t3 = (A[0,0]-A[1,0]) * t
        U[0 * m + 0][col] = 1   # A[0,0]
        U[1 * m + 0][col] = -1  # -A[1,0]
        for j in range(m):
            V[j * p + k][col] = 1  # t = sum_j B[j,k]
        W[k * n + 0][col] = 1    # + for C[0,k]
        W[k * n + 1][col] = -1   # - for C[1,k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
