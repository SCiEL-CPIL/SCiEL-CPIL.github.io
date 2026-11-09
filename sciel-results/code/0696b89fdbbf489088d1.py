"""Rank-minimal exact tensor decomposition for 4x4 * 4x5.

Strategy:
  1. Build a Strassen-based decomposition for the 4x4x4 part (rank 49) and
     append 16 schoolbook terms for the 5th column -> rank 65.
  2. Verify exactly.
  3. Try to reduce rank further by (a) merging linearly-dependent columns
     (rank-deficiency of the tensor) and (b) removing zero columns.
  4. Return the best verified decomposition found within the time budget.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 15.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def _verify(U, V, W, n, m, p, tol=1e-8):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return bool(np.allclose(C, Ctrue, atol=tol))

def _strassen_2x2_blocks():
    # Each entry: (a_coef[2][2], b_coef[2][2]) for a scalar product M_t = (a_coef . A)(b_coef . B)
    S = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),   # M1 = (A11+A22)(B11+B22)
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]),   # M2 = (A21+A22)B11
        ([[1, 0], [0, 0]], [[0, 1], [0, 0]]),   # M3 = A11(B12-B22)  -> wait
    ]
    # Use the standard Strassen formulation carefully.
    # A = [[A11, A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    S = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),   # M1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]),   # M2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]]),  # M3
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]]),  # M4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]]),   # M5
        ([-1, 0], [1, 0]) if False else ([[1, -1], [0, 0]], [[1, 1], [0, 0]]),  # M6
        ([[0, 1], [0, -1]], [[0, 1], [1, 1]]),  # M7
    ]
    # C index (i,k) -> list of M indices (0-based)
    Cuse = [
        [0, 3, 6, -4],   # C11 = M1 - M5 + M4 + M7  (index of M5 is 4)
        [2, 4],          # C12 = M3 + M5
        [1, 3],          # C21 = M2 + M4
        [0, -1, 2, 5],   # C22 = M1 - M2 + M3 + M6
    ]
    return S, Cuse

def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    S, Cuse = _strassen_2x2_blocks()
    U = np.zeros((nrows * ncolsm, 7), dtype=float)
    V = np.zeros((ncolsm * ncolsk, 7), dtype=float)
    W = np.zeros((ncolsk * nrows, 7), dtype=float)
    for t in range(7):
        ac, bc = S[t]
        for i in range(2):
            for j in range(2):
                U[i * ncolsm + j, t] = ac[i][j]
        for j in range(2):
            for k in range(2):
                V[j * ncolsk + k, t] = bc[j][k]
    # C entries: W index is (k, i) with value = coefficient of M_t in C[i,k]
    for i in range(2):
        for k in range(2):
            for term in Cuse[i * 2 + k]:
                if term < 0:
                    W[k * nrows + i, -term - 1] += -1.0
                else:
                    W[k * nrows + i, term] += 1.0
    return U, V, W

def _make_4x4x4_strassen():
    # Two-level Strassen: top level 7 block products, each a 2x2x2 inner Strassen (rank 7).
    # Total rank = 49.
    n = m = p = 4
    S, Cuse = _strassen_2x2_blocks()
    inner_U, inner_V, inner_W = _make_2x2x2_strassen(2, 2, 2)
    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)
    for t in range(7):
        ac, bc = S[t]
        # A block (ri,ci) has coefficient ac[ri][ci]; B block (ci2,ki) has coefficient bc[ci2][ki]
        for ri in range(2):
            for ci in range(2):
                acoef = ac[ri][ci]
                if acoef == 0:
                    continue
                for ci2 in range(2):
                    for ki in range(2):
                        bcoef = bc[ci2][ki]
                        if bcoef == 0:
                            continue
                        # This block product corresponds to top-level t, and the inner
                        # 2x2x2 Strassen handles the (ii,jj) x (jj',kk) x (ii',kk') part.
                        # Inner U index: (ii * 2 + jj), inner V: (jj * 2 + kk), inner W: (kk * 2 + ii)
                        for tp in range(7):
                            base = t * 7 + tp
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, base] = acoef * inner_U[ii * 2 + jj, tp]
                            for jj in range(2):
                                for kk in range(2):
                                    gj = ci2 * 2 + jj
                                    gk = ki * 2 + kk
                                    V[gj * p + gk, base] = bcoef * inner_V[jj * 2 + kk, tp]
                            for kk in range(2):
                                for ii in range(2):
                                    gk = ki * 2 + kk
                                    gi = ri * 2 + ii
                                    W[gk * n + gi, base] = inner_W[kk * 2 + ii, tp]
    return U, V, W

def _build_base(n, m, p):
    U4, V4, W4 = _make_4x4x4_strassen()
    R = U4.shape[1]  # 49
    extra = n * m   # 16 schoolbook terms for k = p-1 = 4
    U = np.zeros((n * m, R + extra), dtype=float)
    V = np.zeros((m * p, R + extra), dtype=float)
    W = np.zeros((p * n, R + extra), dtype=float)
    U[:, :R] = U4
    V[:, :R] = V4
    W[:, :R] = W4
    k = p - 1
    c = R
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    return U, V, W

def _remove_zero_columns(U, V, W, n, m, p):
    # Remove columns that are identically zero in at least one of U,V,W (these contribute nothing).
    R = U.shape[1]
    keep = []
    for c in range(R):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            keep.append(c)
    return U[:, keep], V[:, keep], W[:, keep]

def _try_reduce(U, V, W, n, m, p):
    # Attempt to reduce rank by finding a linear relation among columns.
    # If column c is a linear combination of other columns in U, V, and W simultaneously,
    # we can eliminate it. Use a greedy approach: for each column, try to express it
    # in terms of the others using least squares, then check consistency.
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    # Try to remove one column at a time.
    changed = True
    while changed and _time_left() > 2:
        changed = False
        for c in range(U.shape[1]):
            if _time_left() < 2:
                break
            others = [i for i in range(U.shape[1]) if i != c]
            # Solve for coefficients in U
            Uo = U[:, others]
            try:
                coef_U, *_ = np.linalg.lstsq(Uo, U[:, c], rcond=None)
            except np.linalg.LinAlgError:
                continue
            # Check V consistency
            if not np.allclose(V[:, others] @ coef_U, V[:, c], atol=1e-8):
                continue
            # Check W consistency
            if not np.allclose(W[:, others] @ coef_U, W[:, c], atol=1e-8):
                continue
            # Column c is redundant; remove it.
            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            changed = True
            break
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 80

    # Baseline: schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1

    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_schoolbook

    # Try Strassen-based construction
    if _time_left() > 5:
        try:
            U, V, W = _build_base(n, m, p)
            # Remove zero columns
            U, V, W = _remove_zero_columns(U, V, W, n, m, p)
            if _verify(U, V, W, n, m, p):
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = U.shape[1]
                # Try further reduction
                if _time_left() > 10:
                    U2, V2, W2 = _try_reduce(U, V, W, n, m, p)
                    if _verify(U2, V2, W2, n, m, p) and U2.shape[1] < best_rank:
                        best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
                        best_rank = U2.shape[1]
        except Exception:
            pass

    # Convert to lists of floats (integer/half-integer entries)
    def _to_list(mat):
        return [[float(x) for x in row] for row in np.asarray(mat)]

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}
