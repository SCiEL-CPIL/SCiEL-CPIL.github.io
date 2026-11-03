"""Rank-66 exact tensor decomposition for 3x4 by 4x6 matrix multiplication.

Partition strategy:
  A (3x4): rows split 1+2, cols split 2+2
  B (4x6): rows split 2+2, cols split 2+2+2
  C (3x6): each block row uses two column blocks of A times two row blocks of B

Block ranks:
  1x2 * 2x2  -> rank 4 (schoolbook)
  2x2 * 2x2  -> rank 7 (Strassen)

Total: 6 * (rank 4) + 6 * (rank 7) = 24 + 42 = 66
"""
import time
import numpy as np

def construct():
    _t0 = time.time()
    _deadline = 145.0  # safety margin before 150s kill
    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    RANK = 66
    _pruned = False  # set True if a redundant column is found
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    # Indexing:
    #   A_{i,k}  -> U row  i * m + k       (i in [0,n), k in [0,m))
    #   B_{k,j}  -> V row  k * p + j       (k in [0,m), j in [0,p))
    #   C_{i,j}  -> W row  j * n + i       (j in [0,p), i in [0,n))  (trace convention)

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook).
        A[ar:ar+1, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr, cc], C[cr, cc+1]
        """
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        # M1 = A0 * B00 -> C0
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        # M2 = A1 * B10 -> C0
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        # M3 = A0 * B01 -> C1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        # M4 = A1 * B11 -> C1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen).
        A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr:cr+2, cc:cc+2]
        Standard Strassen:
          M1 = (A00+A11)(B00+B11)   -> contributes to C00, C11
          M2 = (A10+A11)B00         -> C10 += M2, C11 -= M2
          M3 = A00(B01-B11)         -> C01 += M3, C11 += M3
          M4 = A11(B10-B00)         -> C00 += M4, C10 += M4
          M5 = (A00+A01)B11         -> C00 -= M5, C01 += M5
          M6 = (A10-A00)(B00+B01)   -> C11 += M6
          M7 = (A01-A11)(B10+B11)   -> C00 += M7
        """
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # ---- Build the 66-term decomposition ----
    # C has 3 rows × 6 columns.
    # Split A rows: row 0 (1 row), rows 1-2 (2 rows)
    # Split A cols: cols 0-1 (2 cols), cols 2-3 (2 cols)
    # Split B rows: rows 0-1 (2 rows), rows 2-3 (2 rows)
    # Split B cols: cols 0-1, 2-3, 4-5 (three blocks of 2)
    #
    # For each C column-block j (0,1,2) and each A-row-group r,
    # the contribution to C is:
    #   sum over A-col-group ac of sum over B-row-group br:
    #     A_rowgroup_r, colgroup_ac  *  B_rowgroup_br, colgroup_j
    #
    # A row group 0 (row 0) -> 1x2 * 2x2 blocks (rank 4)
    # A row group 1 (rows 1-2) -> 2x2 * 2x2 blocks (rank 7)

    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # --- Row 0 of C (A row 0, 1x2 blocks) ---
        # A[0, 0:2] * B[0:2, bc:bc+2]  (1x2 * 2x2, rank 4)
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        # A[0, 2:4] * B[2:4, bc:bc+2]  (1x2 * 2x2, rank 4)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # --- Rows 1-2 of C (A rows 1:3, 2x2 blocks) ---
        # A[1:3, 0:2] * B[0:2, bc:bc+2]  (2x2 * 2x2, rank 7, Strassen)
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        # A[1:3, 2:4] * B[2:4, bc:bc+2]  (2x2 * 2x2, rank 7, Strassen)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # Safety: verify no term exceeded the allocated rank
    for r in range(n * m):
        for c in range(col):
            assert abs(U[r][c]) <= 1, f"U[{r}][{c}] = {U[r][c]}"
    for r in range(m * p):
        for c in range(col):
            assert abs(V[r][c]) <= 1, f"V[{r}][{c}] = {V[r][c]}"
    for r in range(p * n):
        for c in range(col):
            assert abs(W[r][c]) <= 1, f"W[{r}][{c}] = {W[r][c]}"

    # Deadline guard
    if time.time() > _deadline:
        pass  # We're already done; this is just a safety check

    # --- Support pruning: check if any of the 66 rank-1 columns is
    #     linearly dependent on the remaining 65 via the tensor equation.
    #     The multiplication tensor T[i,k,j] = delta[i,k,j] (1 iff
    #     C_{i,j} = sum_k A_{i,k} B_{k,j}) must equal
    #     sum_r U[i,k,r] * V[k,j,r] * W[j,n+i,r].
    #     We form a (729 x 66) matrix M where M[flat_idx, r] =
    #     U[i,k,r]*V[k,j,r]*W[j,n+i,r] and check if any column is
    #     in the span of the others.
    if time.time() < _deadline - 5:
        T_dim = n * m * p  # 729
        M = np.zeros((T_dim, RANK), dtype=np.float64)
        for r in range(RANK):
            u_col = [U[i][r] for i in range(n * m)]
            v_col = [V[i][r] for i in range(m * p)]
            w_col = [W[i][r] for i in range(p * n)]
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        idx = (i * m + k) * p + j
                        M[idx, r] = u_col[i * m + k] * v_col[k * p + j] * w_col[j * n + i]

        # For each column c, check if M[:,c] is in span(M[:, all except c])
        # Equivalent: rank(M) < RANK means some column is redundant.
        rank_full = np.linalg.matrix_rank(M, tol=1e-10)
        if rank_full < RANK:
            # Find which column(s) are redundant
            for c in range(RANK):
                M_reduced = np.delete(M, c, axis=1)
                rank_reduced = np.linalg.matrix_rank(M_reduced, tol=1e-10)
                if rank_reduced == rank_full:
                    # Column c is redundant; remove it
                    for i in range(n * m):
                        U[i].pop(c)
                    for i in range(m * p):
                        V[i].pop(c)
                    for i in range(p * n):
                        W[i].pop(c)
                    RANK -= 1
                    _pruned = True
                    break

    return {"U": U, "V": V, "W": W}
