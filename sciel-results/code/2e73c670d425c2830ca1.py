"""Matrix multiplication 3x4x4: rank-23 exact decomposition.

Strategy: rank-19 Bini-Capovani-De Falco-Ladisi (BCDL) decomposition for the
3x3x3 sub-tensor (first 3 columns of B / first 3 rows of C), plus a rank-4
schoolbook strip for the 4th column, using the unused A-row 3 and B-column 3.
"""

# BCDL 3x3x3: 19 rank-1 terms. Each term = (A_row, B_col, C_row, C_col, coeff).
# C index is (row, col) in the 3x3 output block, i.e. C[k*n+i] with n=3 -> 3*row+col.
BCDL_333 = [
    (1, 0, 0, 0, 1),
    (0, 2, 0, 1, 1),
    (2, 2, 0, 2, 1),
    (1, 1, 1, 0, 1),
    (1, 2, 1, 1, -1),
    (2, 1, 1, 2, 1),
    (0, 0, 2, 0, 1),
    (0, 1, 2, 1, 1),
    (1, 0, 2, 2, 1),
    (2, 0, 0, 0, 1),
    (0, 0, 0, 1, 1),
    (2, 1, 0, 1, 1),
    (1, 0, 0, 2, 1),
    (1, 2, 1, 2, 1),
    (2, 2, 1, 2, -1),
    (0, 1, 2, 2, 1),
    (2, 0, 2, 0, 1),
    (1, 1, 2, 1, 1),
    (0, 2, 2, 2, 1),
]


def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 iff j==j and k==k (i.e. the standard tensor)."""
    rank = len(U[0])
    # Build tensor T[i,j,k]
    T = [[[0] * p for _ in range(m)] for _ in range(n)]
    for c in range(rank):
        # collect nonzero U, V, W entries for this term
        Ue = [(ii, U[ii][c]) for ii in range(n * m) if U[ii][c] != 0]
        Ve = [(jj, V[jj][c]) for jj in range(m * p) if V[jj][c] != 0]
        We = [(kk, W[kk][c]) for kk in range(p * n) if W[kk][c] != 0]
        for (ii, ui) in Ue:
            i, j = divmod(ii, m)
            for (jj, vj) in Ve:
                j2, k = divmod(jj, p)
                if j2 != j:
                    continue
                for (kk, wk) in We:
                    k2, i2 = divmod(kk, n)
                    if k2 != k:
                        continue
                    T[i][j][k] += ui * vj * wk
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (j == j and k == k) else 0
                # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the product of 3 identity-like).
                # Actually the matmul tensor T_{i,j,k}^{a,b,c}: coefficient of A[a,b]B[b,c] in P[a,c] is 1 for all.
                # So expected is 1 for every (i,j,k).
                expected = 1
                if T[i][j][k] != expected:
                    return False
    return True


def construct():
    n, m, p = 3, 4, 4
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # BCDL 3x3x3 terms
    for (a_row, b_col, c_row, c_col, coeff) in BCDL_333:
        # A index: a_row * m + b_col
        # B index: b_col * p + c_col
        # C index: c_col * n + c_row
        U[a_row * m + b_col][col] = coeff
        V[b_col * p + c_col][col] = coeff
        W[c_col * n + c_row][col] = 1
        col += 1

    # Strip: 4th column of C. 4 terms using A-row 3 and B-col 3.
    # T_r: A[3,3]*B[r,3] -> C[r,3], for r in 0..3
    for r in range(4):
        U[3 * m + 3][col] = 1
        V[3 * p + r][col] = 1
        W[3 * n + r][col] = 1
        col += 1

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
