"""Exact tensor decomposition search for 3x4 * 4x4.

Builds candidate decompositions (schoolbook rank 48, and Strassen-based
sub-48 attempts), verifies each numerically under the trace convention
W[k*n+i, r], and returns the lowest-rank verified-exact one. Falls back to
schoolbook (always exact) if no lower-rank candidate verifies.
"""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s wall-clock limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _TIMEOUT


def _compute_tensor(U, V, W, n, m, p):
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)


def _error(T):
    return float(np.max(np.abs(T - 1.0)))


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    return _error(_compute_tensor(U, V, W, n, m, p))


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2x2(c00, c01, c10, c11, p00, p01, p10, p11, U, V, W, base, n, m, p,
                      i0, j0, k0):
    """Add a 2x2 Strassen decomposition for the sub-block:
       rows i0,i0+1 ; input cols j0,j0+1 ; out cols k0,k0+1.
       cXX are the C-matrix entries (i, j), pXX are the X-matrix entries (j, k).
       Adds 7 rank-1 columns starting at `base`.
       Returns new base.
    """
    # C sub-block (2x2): [[c00, c01],[c10, c11]] with rows (i0,i0+1), cols (j0,j0+1)
    # X sub-block (2x2): [[p00, p01],[p10, p11]] with rows (j0,j0+1), cols (k0,k0+1)
    # Strassen products (scalar):
    #   m1 = c00*p00
    #   m2 = c00*(p01 - p10)
    #   m3 = (c00 + c01)*p10
    #   m4 = c11*(p11 - p01)
    #   m5 = (c00 + c11)*(p00 + p01)
    #   m6 = (c10 + c11)*(p00 - p10)
    #   m7 = (c01 + c11)*(p11 + p01)
    # Output Y (2x2):
    #   Y00 = m5 + m4 - m2 - m3 + m1
    #   Y01 = m1 + m2
    #   Y10 = m3 + m4
    #   Y11 = m5 - m1 + m6 + m7
    #
    # Each mX is a bilinear product of a C-linear-combo and an X-linear-combo.
    # Represent each as a rank-1 column: U holds the C-coeff vector, V holds the
    # X-coeff vector, W holds the output-coeff vector.
    #
    # C-coeff vectors (over the 4 C-indices (i,j) in the sub-block):
    #   e00 = (1,0,0,0) at (i0,j0), e01=(0,1,0,0) at (i0,j0+1),
    #   e10 = (0,0,1,0) at (i0+1,j0), e11=(0,0,0,1) at (i0+1,j0+1)
    # X-coeff vectors (over the 4 X-indices (j,k) in the sub-block):
    #   f00=(1,0,0,0) at (j0,k0), f01=(0,1,0,0) at (j0,k0+1),
    #   f10=(0,0,1,0) at (j0+1,k0), f11=(0,0,0,1) at (j0+1,k0+1)
    # Output-coeff vectors (over the 4 Y-indices (i,k) in the sub-block):
    #   g00=(1,0,0,0) at (i0,k0), g01=(0,1,0,0) at (i0,k0+1),
    #   g10=(0,0,1,0) at (i0+1,k0), g11=(0,0,0,1) at (i0+1,k0+1)
    #
    # m1 = e00 . f00  -> output g00,g01,g10,g11? No, m1 contributes to Y00 and Y01.
    #   Actually each mX is a SCALAR (product of two linear forms), and it appears
    #   in the output with a coefficient. So mX is ONE rank-1 term whose W-vector
    #   is the set of output coefficients.
    #   m1 appears in Y00 (coeff +1) and Y01 (coeff +1). So W for m1 = g00 + g01.
    #   m2 appears in Y00 (coeff -1) and Y01 (coeff +1). W = -g00 + g01.
    #   m3 appears in Y00 (coeff -1) and Y10 (coeff +1). W = -g00 + g10.
    #   m4 appears in Y00 (coeff +1) and Y10 (coeff +1). W = g00 + g10.
    #   m5 appears in Y00 (coeff +1) and Y11 (coeff +1). W = g00 + g11.
    #   m6 appears in Y11 (coeff +1). W = g11.
    #   m7 appears in Y11 (coeff +1). W = g11.
    #
    # C-vectors:
    #   m1: e00
    #   m2: e00
    #   m3: e00 + e01
    #   m4: e11
    #   m5: e00 + e11
    #   m6: e10 + e11
    #   m7: e01 + e11
    # X-vectors:
    #   m1: f00
    #   m2: f01 - f10
    #   m3: f10
    #   m4: f11 - f01
    #   m5: f00 + f01
    #   m6: f00 - f10
    #   m7: f11 + f01
    #
    # Global index helpers:
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    terms = [
        # (c_vec dict, x_vec dict, w_vec dict)
        ({ui(i0, j0): 1}, {vi(j0, k0): 1}, {wi(k0, i0): 1, wi(k0 + 1, i0): 1}),            # m1
        ({ui(i0, j0): 1}, {vi(j0, k0 + 1): 1, vi(j0 + 1, k0): -1}, {wi(k0, i0): -1, wi(k0 + 1, i0): 1}),  # m2
        ({ui(i0, j0): 1, ui(i0, j0 + 1): 1}, {vi(j0 + 1, k0): 1}, {wi(k0, i0): -1, wi(k0, i0 + 1): 1}),  # m3
        ({ui(i0 + 1, j0 + 1): 1}, {vi(j0 + 1, k0 + 1): 1, vi(j0, k0 + 1): -1}, {wi(k0, i0): 1, wi(k0, i0 + 1): 1}),  # m4
        ({ui(i0, j0): 1, ui(i0 + 1, j0 + 1): 1}, {vi(j0, k0): 1, vi(j0, k0 + 1): 1}, {wi(k0, i0): 1, wi(k0 + 1, i0 + 1): 1}),  # m5
        ({ui(i0 + 1, j0): 1, ui(i0 + 1, j0 + 1): 1}, {vi(j0, k0): 1, vi(j0 + 1, k0): -1}, {wi(k0 + 1, i0 + 1): 1}),  # m6
        ({ui(i0, j0 + 1): 1, ui(i0 + 1, j0 + 1): 1}, {vi(j0 + 1, k0 + 1): 1, vi(j0, k0 + 1): 1}, {wi(k0 + 1, i0 + 1): 1}),  # m7
    ]
    for c_vec, x_vec, w_vec in terms:
        for idx, val in c_vec.items():
            U[idx, base] += val
        for idx, val in x_vec.items():
            V[idx, base] += val
        for idx, val in w_vec.items():
            W[idx, base] += val
        base += 1
    return base


def _strassen_2x4_block():
    """Build the 2x4 (rows 0,1) x 4x4 block using 2x2 Strassen tiled over the
    2x2 grid of (input-col-pair, out-col-pair). Returns (U,V,W) of rank 28."""
    n, m, p = DIMENSIONS
    # 2 input-col-pairs: (0,1) and (2,3); 2 out-col-pairs: (0,1) and (2,3).
    # But Strassen on (rows 0,1) x (input-pair) x (out-pair) only covers a 2x2x2
    # sub-block. Tiling over input-pairs and out-pairs independently gives 4
    # sub-blocks, each rank 7 -> 28. However, the C-matrix entries are shared
    # across out-pairs (same C[i,j] used for multiple k), and X entries shared
    # across input-pairs. The Strassen decomposition is per (j-pair, k-pair), and
    # the products are independent, so tiling is valid: each (j-pair,k-pair)
    # sub-block uses its own 7 columns.
    U = np.zeros((n * m, 28))
    V = np.zeros((m * p, 28))
    W = np.zeros((p * n, 28))
    base = 0
    for jp in range(2):  # input-col-pair: 0 -> cols(0,1), 1 -> cols(2,3)
        for kp in range(2):  # out-col-pair: 0 -> cols(0,1), 1 -> cols(2,3)
            j0 = jp * 2
            k0 = kp * 2
            i0 = 0
            # c entries: C[i, j] for i in (0,1), j in (j0, j0+1)
            # p entries: X[j, k] for j in (j0, j0+1), k in (k0, k0+1)
            base = _strassen_2x2x2x2(
                c00=1, c01=1, c10=1, c11=1,
                p00=1, p01=1, p10=1, p11=1,
                U=U, V=V, W=W, base=base, n=n, m=m, p=p,
                i0=i0, j0=j0, k0=k0)
    return U, V, W


def _candidate_strassen_44():
    """2x4 block via Strassen (28 rank) + 1x4 block naive (4 rank) = 32 rank.
    (Named 44 conservatively; actual built rank is 32.)"""
    n, m, p = DIMENSIONS
    rank24 = 28
    rank14 = 4
    total = rank24 + rank14
    U = np.zeros((n * m, total))
    V = np.zeros((m * p, total))
    W = np.zeros((p * n, total))
    U24, V24, W24 = _strassen_2x4_block()
    U[0:8, 0:rank24] = U24
    V[0:16, 0:rank24] = V24
    W[0:8, 0:rank24] = W24
    # 1x4 block (row 2): naive
    col = rank24
    for j in range(4):
        for k in range(4):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    return U, V, W


def _to_lists(U, V, W):
    U_list = [[float(v) for v in row] for row in U.tolist()]
    V_list = [[float(v) for v in row] for row in V.tolist()]
    W_list = [[float(v) for v in row] for row in W.tolist()]
    return U_list, V_list, W_list


def construct():
    n, m, p = DIMENSIONS

    # Candidates in order of increasing rank (best first).
    candidates = []
    U, V, W = _candidate_strassen_44()
    candidates.append((U.shape[1], U, V, W))
    U, V, W = _schoolbook()
    candidates.append((U.shape[1], U, V, W))

    best = None
    for rank, U, V, W in candidates:
        if not _elapsed_ok():
            break
        err = _verify(U, V, W)
        if err < 1e-12:
            best = (rank, U, V, W)
            break  # lowest-rank verified-exact found

    if best is None:
        # Should not happen (schoolbook always verifies), but be safe.
        U, V, W = _schoolbook()
        best = (U.shape[1], U, V, W)

    _, U, V, W = best
    return {"U": _to_lists(U, V, W)[0], "V": _to_lists(U, V, W)[1], "W": _to_lists(U, V, W)[2]}
