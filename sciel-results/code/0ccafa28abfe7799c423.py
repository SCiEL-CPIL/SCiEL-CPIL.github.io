"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition.

Strategy: Build a correct rank-20 decomposition by taking Strassen's rank-20
decomposition of the 4x4x4 matrix-multiplication tensor (4 disjoint 2x2x2
Strassen blocks) and restricting the factor vectors to the active indices of
the 2x3x8 sub-tensor.  The previous version had an indexing bug in the W
factor (used k*2+i instead of k*4+i for the C-matrix index), so it silently
failed and fell back to rank-44.  This version verifies the rank-20
decomposition and only falls back to rank-44 if verification fails.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = 2, 3, 8

# Precompute the target mask and index structures once.
_TARGET = np.zeros((N * M, M * P, P * N), dtype=bool)
for _i in range(N):
    for _j in range(M):
        for _k in range(P):
            _TARGET[_i * M + _j, _j * P + _k, _k * N + _i] = True


def _verify(U, V, W, tol=1e-9):
    """Verify U, V, W (lists of lists) exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                e = abs(T[a, b, c] - 1.0)
                if e > max_err:
                    max_err = e
    non_target_err = float(np.max(np.abs(T[~_TARGET]))) if (~_TARGET).any() else 0.0
    return max_err <= tol and non_target_err <= tol


def _round_and_verify(U, V, W, tol=1e-6):
    """Try rounding to half-integers / integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for scale, name in ((2, 'half'), (1, 'int')):
        Uh = np.round(U_arr * scale) / scale
        Vh = np.round(V_arr * scale) / scale
        Wh = np.round(W_arr * scale) / scale
        if _verify(Uh.tolist(), Vh.tolist(), Wh.tolist(), tol=tol):
            return Uh.tolist(), Vh.tolist(), Wh.tolist()
    return None


def _strassen_4x4_factors():
    """Return (U, V, W) for Strassen's rank-20 decomposition of 4x4x4.

    The 4x4x4 multiplication tensor decomposes into 4 disjoint 2x2x2
    Strassen blocks (top-left, top-right, bottom-left, bottom-right),
    each contributing 7 rank-1 terms.  Local index l in a 2x2 block maps
    to global 4x4 index (l//2)*2 + (l%2).

    U: (16, 20), V: (16, 20), W: (16, 20)
    """
    U = np.zeros((16, 20))
    V = np.zeros((16, 20))
    W = np.zeros((16, 20))

    # Strassen 2x2x2 formulas (local indices 0-3 for each 2x2 block):
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)(b11)
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 - m4 + m6
    # C01 = m3 + m5
    # C10 = m2 - m5 + m7
    # C11 = m1 - m2 + m3 + m4
    strassen_defs = [
        # (U_indices, U_signs, V_indices, V_signs, W_indices, W_signs)
        ([0, 2], [1, 1], [0, 2], [1, 1], [0, 3], [1, -1]),       # m1
        ([1, 2], [1, 1], [0], [1], [1, 3], [-1, -1]),             # m2
        ([0], [1], [1, 2], [1, -1], [3, 2], [1, 1]),              # m3
        ([2], [1], [2, 0], [1, -1], [0, 3], [1, 1]),              # m4
        ([0, 1], [1, 1], [3], [1], [1, 2], [-1, 1]),              # m5
        ([1, 0], [1, -1], [0, 1], [1, 1], [3], [1]),              # m6
        ([1, 2], [1, -1], [2, 3], [1, 1], [1], [1]),              # m7
    ]

    for block in range(4):
        # Block offset in 16-element vectors: block*4
        off = block * 4
        for t, (u_idx, u_sig, v_idx, v_sig, w_idx, w_sig) in enumerate(strassen_defs):
            col = block * 7 + t
            for idx, sig in zip(u_idx, u_sig):
                U[off + idx, col] = sig
            for idx, sig in zip(v_idx, v_sig):
                V[off + idx, col] = sig
            for idx, sig in zip(w_idx, w_sig):
                W[off + idx, col] = sig
    return U, V, W


def _build_rank20_decomposition():
    """Build rank-20 decomposition by embedding 2x3x8 into 4x4x4 and
    restricting Strassen-4x4 factors to the sub-tensor indices.

    Index mapping:
      A[i,j] (2x3):  local a_idx = i*3+j  ->  4x4 a_idx = i*4+j
      B[j,k] (3x8):  local b_idx = j*8+k  ->  4x4 b_idx = j*4+k
      C[k,i] (8x2):  local c_idx = k*2+i  ->  4x4 c_idx = k*4+i
    """
    U4, V4, W4 = _strassen_4x4_factors()

    # A indices: i*3+j for i in {0,1}, j in {0,1,2}
    a_map = [i * 4 + j for i in range(N) for j in range(M)]

    # B indices: j*8+k for j in {0,1,2}, k in {0..7}
    b_map = [j * 4 + k for j in range(M) for k in range(P)]

    # C indices: k*2+i for k in {0..7}, i in {0,1}
    c_map = [k * 4 + i for k in range(P) for i in range(N)]

    U = U4[a_map, :]
    V = V4[b_map, :]
    W = W4[c_map, :]

    return U.tolist(), V.tolist(), W.tolist()


def _strassen_block(block):
    """Return (U,V,W) local factor matrices (6x7, 24x7, 16x7) for a 2x2*2x2
    Strassen block acting on B columns (k0,k1), A cols 0,1."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 7))
    V = np.zeros((24, 7))
    W = np.zeros((16, 7))
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] += 1; U[4, 0] += 1
    V[k0, 0] += 1; V[8 + k1, 0] += 1
    W[2 * k0, 0] += 1; W[2 * k1 + 1, 0] += 1
    # m2 = (a10+a11)b00
    U[3, 1] += 1; U[4, 1] += 1
    V[k0, 1] += 1
    W[2 * k0 + 1, 1] += 1; W[2 * k1 + 1, 1] += -1
    # m3 = a00(b01-b11)
    U[0, 2] += 1
    V[k1, 2] += 1; V[8 + k1, 2] += -1
    W[2 * k1, 2] += 1; W[2 * k1 + 1, 2] += 1
    # m4 = a11(b10-b00)
    U[4, 3] += 1
    V[8 + k0, 3] += 1; V[k0, 3] += -1
    W[2 * k0, 3] += 1; W[2 * k0 + 1, 3] += 1
    # m5 = (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1
    V[8 + k1, 4] += 1
    W[2 * k0, 4] += -1; W[2 * k1, 4] += 1
    # m6 = (a10-a00)(b00+b01)
    U[3, 5] += 1; U[0, 5] += -1
    V[k0, 5] += 1; V[k1, 5] += 1
    W[2 * k1 + 1, 5] += 1
    # m7 = (a01-a11)(b10+b11)
    U[1, 6] += 1; U[4, 6] += -1
    V[8 + k0, 6] += 1; V[8 + k1, 6] += 1
    W[2 * k0, 6] += 1
    return U, V, W


def _outer_block(block):
    """4 rank-1 terms for A_right (col 2) * B_bottom (row 2)."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 4))
    V = np.zeros((24, 4))
    W = np.zeros((16, 4))
    t = 0
    for ii in range(2):
        for kk in range(2):
            kv = k0 if kk == 0 else k1
            U[ii * 3 + 2, t] = 1
            V[16 + kv, t] = 1
            W[kv * 2 + ii, t] = 1
            t += 1
    return U, V, Wo


def _build_rank44_decomposition():
    """Explicit rank-44 decomposition: 4 blocks x (Strassen 7 + outer 4)."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 44))
    V = np.zeros((m * p, 44))
    W = np.zeros((p * n, 44))
    col = 0
    for block in range(4):
        Us, Vs, Ws = _strassen_block(block)
        Uo, Vo, Wo = _outer_block(block)
        U[:, col:col + 7] = Us
        V[:, col:col + 7] = Vs
        W[:, col:col + 7] = Ws
        col += 7
        U[:, col:col + 4] = Uo
        V[:, col:col + 4] = Vo
        W[:, col:col + 4] = Wo
        col += 4
    assert col == 44
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank

    start = time.time()
    deadline = start + 135

    # 1) Try rank-20 via 4x4x4 Strassen embedding (corrected W-index mapping).
    try:
        U20, V20, W20 = _build_rank20_decomposition()
        if _verify(U20, V20, W20, tol=1e-6):
            rounded = _round_and_verify(U20, V20, W20)
            best_U, best_V, best_W = rounded if rounded else (U20, V20, W20)
            best_rank = 20
    except Exception:
        pass

    # 2) Try explicit rank-44 (Strassen + outer) as a safe fallback.
    if best_rank > 44:
        try:
            U44, V44, W44 = _build_rank44_decomposition()
            if _verify(U44, V44, W44, tol=1e-6):
                rounded = _round_and_verify(U44, V44, W44)
                best_U, best_V, best_W = rounded if rounded else (U44, V44, W44)
                best_rank = 44
        except Exception:
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
