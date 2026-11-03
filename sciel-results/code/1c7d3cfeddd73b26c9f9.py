"""Rank-189 decomposition for 6x6 matrix multiplication.
Uses 2x2 block Strassen (rank 7) combined with 3x3 schoolbook (rank 27)
subblocks, giving total rank 7 * 27 = 189.
"""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build_3x3_schoolbook(n):
    """Return (U3, V3, W3) for n x n matrix multiplication (rank n^3).
    U3 indexed by (i,j) -> row i*n+j, V3 by (j,k) -> row j*n+k,
    W3 by (k,i) -> row k*n+i.
    """
    rank = n * n * n
    U3 = [[0.0] * rank for _ in range(n * n)]
    V3 = [[0.0] * rank for _ in range(n * n)]
    W3 = [[0.0] * rank for _ in range(n * n)]
    c = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U3[i * n + j][c] = 1.0
                V3[j * n + k][c] = 1.0
                W3[k * n + i][c] = 1.0
                c += 1
    return U3, V3, W3

def _build_2x2_strassen():
    """Return (U2, V2, W2) scalar factors for 2x2 Strassen (rank 7).
    U2 indexed by (i,j), V2 by (j,k), W2 by (k,i).
    Standard Strassen:
      M1 = (A11+A22)*(B11+B22)
      M2 = (A21+A22)*B11
      M3 = A11*(B12-B22)
      M4 = A22*(B21-B11)
      M5 = (A11+A12)*B22
      M6 = (A21-A11)*(B11+B12)
      M7 = (A12-A22)*(B21+B22)
      C11 = M1+M4-M5+M7
      C12 = M3+M5
      C21 = M2+M4
      C22 = M1-M2+M3+M6
    """
    rank = 7
    U2 = [[0.0] * rank for _ in range(4)]  # (i,j)
    V2 = [[0.0] * rank for _ in range(4)]  # (j,k)
    W2 = [[0.0] * rank for _ in range(4)]  # (k,i)

    # M1
    U2[0][0] = 1.0; U2[3][0] = 1.0
    V2[0][0] = 1.0; V2[1][0] = 1.0; V2[2][0] = 1.0; V2[3][0] = 1.0
    W2[0][0] = 1.0; W2[3][0] = 1.0
    # M2
    U2[2][1] = 1.0; U2[3][1] = 1.0
    V2[0][1] = 1.0
    W2[2][1] = 1.0; W2[3][1] = 1.0
    # M3
    U2[0][2] = 1.0
    V2[1][2] = 1.0; V2[3][2] = -1.0
    W2[1][2] = 1.0; W2[3][2] = 1.0
    # M4
    U2[3][3] = 1.0
    V2[2][3] = 1.0; V2[0][3] = -1.0
    W2[0][3] = 1.0; W2[2][3] = 1.0
    # M5
    U2[0][4] = 1.0; U2[1][4] = 1.0
    V2[3][4] = 1.0
    W2[1][4] = -1.0; W2[3][4] = 1.0
    # M6
    U2[2][5] = 1.0; U2[0][5] = -1.0
    V2[0][5] = 1.0; V2[1][5] = 1.0
    W2[3][5] = 1.0
    # M7
    U2[1][6] = 1.0; U2[3][6] = -1.0
    V2[2][6] = 1.0; V2[3][6] = 1.0
    W2[0][6] = 1.0
    return U2, V2, W2

def _kron_combine(U2, V2, W2, U3, V3, W3, n):
    """Combine 2x2 block-level factors with 3x3 subblock factors.
    Block (I,J) occupies rows I*3..I*3+2 in each of the 6-row index spaces.
    Full index: row = block_index * 3 + sub_index.
    """
    n_blocks = 2
    rank_total = 7 * (n * n * n)
    N = n_blocks * n  # 6
    U = [[0.0] * rank_total for _ in range(N * N)]
    V = [[0.0] * rank_total for _ in range(N * N)]
    W = [[0.0] * rank_total for _ in range(N * N)]
    for a in range(7):
        for t in range(n * n * n):
            col = a * (n * n * n) + t
            # U: row (I,J) in 2x2-block space -> full row I*n + J*n + sub_i*n + sub_j
            for I in range(n_blocks):
                for J in range(n_blocks):
                    u2val = U2[I * n_blocks + J][a]
                    if u2val == 0.0:
                        continue
                    for ii in range(n):
                        for jj in range(n):
                            u3val = U3[ii * n + jj][t]
                            if u3val == 0.0:
                                continue
                            full_i = I * n + ii
                            full_j = J * n + jj
                            U[full_i * N + full_j][col] += u2val * u3val
            # V: (J,K) block space
            for J in range(n_blocks):
                for K in range(n_blocks):
                    v2val = V2[J * n_blocks + K][a]
                    if v2val == 0.0:
                        continue
                    for jj in range(n):
                        for kk in range(n):
                            v3val = V3[jj * n + kk][t]
                            if v3val == 0.0:
                                continue
                            full_j = J * n + jj
                            full_k = K * n + kk
                            V[full_j * N + full_k][col] += v2val * v3val
            # W: (K,I) block space
            for K in range(n_blocks):
                for I in range(n_blocks):
                    w2val = W2[K * n_blocks + I][a]
                    if w2val == 0.0:
                        continue
                    for kk in range(n):
                        for ii in range(n):
                            w3val = W3[kk * n + ii][t]
                            if w3val == 0.0:
                                continue
                            full_k = K * n + kk
                            full_i = I * n + ii
                            W[full_k * N + full_i][col] += w2val * w3val
    return U, V, W

def construct():
    n = 3
    U3, V3, W3 = _build_3x3_schoolbook(n)
    U2, V2, W2 = _build_2x2_strassen()
    U, V, W = _kron_combine(U2, V2, W2, U3, V3, W3, n)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
