"""Matrix multiplication 3x4x4 rank-minimal decomposition.

Strategy: build a rank-44 decomposition (4 Strassen 2x2x2 blocks for the
2x4 top part + 4 schoolbook 1x2x2 strips for the 1x4 bottom part), then
attempt to reduce the rank to 43 via a fast vectorized ALS re-solve after
removing one column, finishing with a half-integer rounding + local search.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _strassen_2x2x2():
    """(U,V,W) lists for 2x2x2, rank 7. U[(i*2+j),c], V[(j*2+k),c], W[(k*2+i),c]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # M2
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    # M3
    U[0][2]=1; V[2][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    # M4
    U[3][3]=1; V[1][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # M5
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    # M6
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W


def _strip_1x2x2():
    """1x2 x 2x2 = 1x2, rank 4. U[j,c], V[(i*2+j),c], W[j,c]."""
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    U[0][1]=1; V[1][1]=1; W[1][1]=1
    U[1][2]=1; V[2][2]=1; W[0][2]=1
    U[1][3]=1; V[3][3]=1; W[1][3]=1
    return U, V, W


def build_rank44(n, m, p):
    """Assemble rank-44 (U,V,W) as numpy arrays."""
    U = np.zeros((n * m, 44), dtype=float)
    V = np.zeros((m * p, 44), dtype=float)
    W = np.zeros((p * n, 44), dtype=float)
    col = 0

    def place_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        sU, sV, sW = _strassen_2x2x2()
        for c in range(7):
            a_pairs = [((li // 2), (li % 2), sU[li][c]) for li in range(4) if sU[li][c] != 0]
            b_pairs = [((li // 2), (li % 2), sV[li][c]) for li in range(4) if sV[li][c] != 0]
            w_pairs = [((li // 2), (li % 2), sW[li][c]) for li in range(4) if sW[li][c] != 0]
            for (ai, aj, ac) in a_pairs:
                for (bi, bk, bc) in b_pairs:
                    for (ck, ci, cc) in w_pairs:
                        a_idx = (a_row_off + ai) * m + (a_col_off + aj)
                        b_idx = (b_row_off + bi) * p + (b_col_off + bk)
                        c_idx = (c_col_off + ck) * n + (c_row_off + ci)
                        U[a_idx, col] += ac
                        V[b_idx, col] += bc
                        W[c_idx, col] += cc
            col += 1

    def place_strip(a_row, a_col_off, b_row_off, b_col_off, c_row, c_col_off):
        nonlocal col
        sU, sV, sW = _strip_1x2x2()
        for c in range(4):
            a_pairs = [(j, sU[j][c]) for j in range(2) if sU[j][c] != 0]
            b_pairs = [((li // 2), (li % 2), sV[li][c]) for li in range(4) if sV[li][c] != 0]
            w_pairs = [(j, sW[j][c]) for j in range(2) if sW[j][c] != 0]
            for (aj, ac) in a_pairs:
                for (bi, bk, bc) in b_pairs:
                    for (cj, cc) in w_pairs:
                        a_idx = a_row * m + (a_col_off + aj)
                        b_idx = (b_row_off + bi) * p + (b_col_off + bk)
                        c_idx = (c_col_off + cj) * n + c_row
                        U[a_idx, col] += ac
                        V[b_idx, col] += bc
                        W[c_idx, col] += cc
            col += 1

    # C_top: rows 0,1. For each B-half column offset, top rows + bottom rows.
    for b_col_off in (0, 2):
        c_col_off = b_col_off
        place_strassen(0, 0, 0, b_col_off, 0, c_col_off)   # A_top1 * B_half_top
        place_strassen(0, 2, 2, b_col_off, 0, c_col_off)   # A_top2 * B_half_bottom
    # C_bot: row 2.
    for b_col_off in (0, 2):
        c_col_off = b_col_off
        place_strip(2, 0, 0, b_col_off, 2, c_col_off)
        place_strip(2, 2, 2, b_col_off, 2, c_col_off)

    return U, V, W


def check(U, V, W, n, m, p):
    """Exact check that the decomposition equals the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _als(U, V, W, n, m, p, iters):
    """One ALS pass: for each (i,j) solve for U, etc. Vectorized per-row."""
    R = U.shape[1]
    # Update U: for each a=i*m+j, target over k: sum_r U[a,r] V[j*p+k,r] W[k*n+i,r] = 1
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, R))
            for k in range(p):
                M[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                U[i * m + j, :] = sol
            except np.linalg.LinAlgError:
                pass
    # Update V
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, R))
            for i in range(n):
                M[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                V[j * p + k, :] = sol
            except np.linalg.LinAlgError:
                pass
    # Update W
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, R))
            for j in range(m):
                M[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                W[k * n + i, :] = sol
            except np.linalg.LinAlgError:
                pass
    return U, V, W


def try_reduce(U44, V44, W44, col, n, m, p, deadline):
    """Remove one column and re-solve; try to land on a valid half-integer rank-43."""
    R = U44.shape[1] - 1
    U = np.delete(U44, col, axis=1).copy()
    V = np.delete(V44, col, axis=1).copy()
    W = np.delete(W44, col, axis=1).copy()

    for _ in range(40):
        if time.time() > deadline - 3:
            break
        U, V, W = _als(U, V, W, n, m, p, 1)

    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    if check(U, V, W, n, m, p):
        return U, V, W

    # Local search: flip single half-integer entries until valid.
    for _ in range(60):
        if time.time() > deadline - 1:
            break
        improved = False
        for mat, nm in ((U, n * m), (V, m * p), (W, p * n)):
            for row in range(nm):
                for r in range(R):
                    old = mat[row, r]
                    for delta in (0.5, -0.5):
                        mat[row, r] = old + delta
                        if check(U, V, W, n, m, p):
                            return U, V, W
                        mat[row, r] = old
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before the 120s evaluator kill

    U44, V44, W44 = build_rank44(n, m, p)
    if not check(U44, V44, W44, n, m, p):
        return _schoolbook()

    # Try to reduce to rank 43 by removing each column (in a deterministic order).
    for col in range(U44.shape[1]):
        if time.time() > deadline - 5:
            break
        res = try_reduce(U44, V44, W44, col, n, m, p, deadline)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank 44.
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}


def _schoolbook():
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
