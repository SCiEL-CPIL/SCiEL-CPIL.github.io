import time
import numpy as np

DIMENSIONS = [4, 4, 5]

DEADLINE_FRAC = 0.90          # return before ~150s wall-clock
START = time.time()

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC

# ----------------------------------------------------------------------
# Exact schoolbook decomposition (rank n*m*p = 80). Guaranteed correct.
# ----------------------------------------------------------------------
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


# ----------------------------------------------------------------------
# Verify a candidate (U,V,W) exactly reproduces matmul C = A @ B.
# C[i,k] = sum_r U[i,r]*V[k,r]*W[k*n+i,r]   (official trace convention)
# ----------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-6):
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
    # build C from tensor
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)


# ----------------------------------------------------------------------
# 2x2x2 Strassen block (rank 7) used at the lowest recursion level.
# Returns coefficient tables as lists over indices.
# ----------------------------------------------------------------------
def _strassen_2x2_blocks():
    # Standard Strassen for 2x2x2.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    # Represent as (a_coef[2][2], b_coef[2][2])
    S = [
        ([[1, 0], [0, 1]], [[1, 1], [1, 1]]),   # M1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]),   # M2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]]),  # M3
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]]),  # M4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]]),   # M5
        ([[1, 0], [-1, 0]], [[1, 1], [0, 0]]),  # M6
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]]),  # M7
    ]
    # C index: C[i*2+k] lists which M's contribute
    Cuse = [
        [0, 3, 4, 6],   # c00 = M1 + M4 - M5 + M7
        [2, 4],          # c01 = M3 + M5
        [1, 3],          # c10 = M2 + M4
        [0, 1, 2, 5],   # c11 = M1 - M2 + M3 + M6
    ]
    # Signs for each M contribution to each C element
    Csign = [
        [1, 1, -1, 1],   # c00
        [1, 1],           # c01
        [1, 1],           # c10
        [1, -1, 1, 1],   # c11
    ]
    return S, Cuse, Csign


def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    """Build U(nrows*ncolsm,7) V(ncolsm*ncolsk,7) W(ncolsk*nrows,7) for 2x2x2."""
    S, Cuse, Csign = _strassen_2x2_blocks()
    U = np.zeros((nrows * ncolsm, 7), dtype=float)
    V = np.zeros((ncolsm * ncolsk, 7), dtype=float)
    W = np.zeros((ncolsk * nrows, 7), dtype=float)
    for t, (ac, bc) in enumerate(S):
        for i in range(2):
            for j in range(2):
                U[i * ncolsm + j, t] = ac[i][j]
        for j in range(2):
            for k in range(2):
                V[j * ncolsk + k, t] = bc[j][k]
    # C index: row i in 0..1, col k in 0..1, W index is k*nrows+i
    for i in range(2):
        for k in range(2):
            idx = i * 2 + k
            for mi, t in enumerate(Cuse[idx]):
                W[k * nrows + i, t] = Csign[idx][mi]
    return U, V, W


# ----------------------------------------------------------------------
# Recursively build the 4x4x4 Strassen block via block decomposition,
# then append 4 extra columns (one per extra k) with schoolbook on A-rows.
# ----------------------------------------------------------------------
def _strassen_444_and_append(n, m, p):
    # A is n x m = 4 x 4, B is m x p = 4 x 5.
    # Use Strassen for the 4x4x4 part (k=0..3), then schoolbook for k=4.
    U4, V4, W4 = _make_4x4x4_strassen()
    R = U4.shape[1]
    extra = n * m   # 16 columns for the 5th output column
    U = np.zeros((n * m, R + extra), dtype=float)
    V = np.zeros((m * p, R + extra), dtype=float)
    W = np.zeros((p * n, R + extra), dtype=float)
    U[:, :R] = U4
    V[:, :R] = V4
    W[:, :R] = W4
    k = 4
    c = R
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    return U, V, W


# Build 4x4x4 by 2-level block Strassen: top level 2x2 blocks each of size 2x2.
# Top-level rank 7; each top-level multiply M_t is (2x2 block of A)*(2x2 block of B)
# which is itself 2x2x2 -> use inner Strassen (rank 7). Total 7*7=49.
def _make_4x4x4_strassen():
    S, Cuse, Csign = _strassen_2x2_blocks()
    n = m = p = 4
    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)
    inner_U, inner_V, inner_W = _make_2x2x2_strassen(2, 2, 2)
    for t in range(7):
        ac = S[t][0]; bc = S[t][1]
        for tp in range(7):
            col = t * 7 + tp
            # For each (ri, ci) block of A and (ci2, ki) block of B:
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
                            # U: for all (ii, jj) in the (ri, ci) block
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, col] += acoef * inner_U[ii * 2 + jj, tp]
                            # V: for all (jj2, kk) in the (ci2, ki) block
                            for jj2 in range(2):
                                for kk in range(2):
                                    gj = ci2 * 2 + jj2
                                    gk = ki * 2 + kk
                                    V[gj * p + gk, col] += bcoef * inner_V[jj2 * 2 + kk, tp]
            # W: top-level C combination with signs
            for ri in range(2):
                for ki in range(2):
                    idx = ri * 2 + ki
                    for mi, t_inner in enumerate(Cuse[idx]):
                        if t_inner == tp:
                            sign = Csign[idx][mi]
                            for ii in range(2):
                                for kk in range(2):
                                    gi = ri * 2 + ii
                                    gk = ki * 2 + kk
                                    W[gk * n + gi, col] += sign * inner_W[kk * 2 + ii, tp]
    return U, V, W


# ----------------------------------------------------------------------
# Final: build candidates, verify, pick minimal verified rank; fallback schoolbook.
# ----------------------------------------------------------------------
def construct():
    n, m, p = DIMENSIONS
    best = None
    # Candidate 1: schoolbook (always correct, rank 80)
    Us, Vs, Ws = _schoolbook(n, m, p)
    if _verify(Us, Vs, Ws, n, m, p):
        best = (Us, Vs, Ws, Us.shape[1])
    # Candidate 2: 4x4x4 Strassen (49) + 16 extra schoolbook = rank 65
    if not _deadline():
        try:
            U2, V2, W2 = _strassen_444_and_append(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                if best is None or U2.shape[1] < best[3]:
                    best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass
    if best is None:
        best = (Us, Vs, Ws, Us.shape[1])
    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
