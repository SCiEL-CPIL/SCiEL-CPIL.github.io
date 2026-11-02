"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search.

Strategy: Decompose into four 2x3x2 blocks. For each block, use Strassen-7 for
the 2x2x2 core plus 4 rank-1 terms for the extra dimension (rank 11 total),
giving rank 44. Then use ALS to search for rank-10 per block (rank 40 total).
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
    return U, V, W


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


def _als_block(rank, block, max_iters=100, deadline=None, seed=None):
    """ALS on a single 2x3x2 block to find a rank-`rank` scheme for the
    local multiplication tensor. Returns (U,V,W) numpy arrays or None."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    U = rng.randn(6, rank) * 0.5
    V = rng.randn(24, rank) * 0.5
    W = rng.randn(16, rank) * 0.5
    k0 = 2 * block
    k1 = 2 * block + 1
    prev = float('inf')
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            return None
        # update U
        for a in range(6):
            Mm = np.zeros((16, rank))
            row = 0
            for j in range(3):
                for k in (k0, k1):
                    Mm[row, :] = V[j * 8 + k, :] * W[k * 2 + (a // 3), :]
                    row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                U[a, :] = x
            except Exception:
                pass
        # update V
        for b in range(24):
            j = b // 8
            k = b % 8
            if k not in (k0, k1):
                continue
            Mm = np.zeros((16, rank))
            row = 0
            for i in range(2):
                for kk in (k0, k1):
                    Mm[row, :] = U[i * 3 + j, :] * W[kk * 2 + i, :]
                    row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                V[b, :] = x
            except Exception:
                pass
        # update W
        for c in range(16):
            kk = c // 2
            i = c % 2
            if kk not in (k0, k1):
                continue
            Mm = np.zeros((16, rank))
            row = 0
            for ii in range(2):
                for j in range(3):
                    Mm[row, :] = U[ii * 3 + j, :] * V[j * 8 + kk, :]
                    row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                W[c, :] = x
            except Exception:
                pass
        # residual
        T = np.zeros((6, 24, 16))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = 0.0
        for i in range(2):
            for j in range(3):
                for k in (k0, k1):
                    e = abs(T[i * 3 + j, j * 8 + k, k * 2 + i] - 1.0)
                    if e > err:
                        err = e
        if err < 1e-9:
            return U, V, W
        if err > prev * 1.5:
            break
        prev = err
    return None


def _build_rank40_decomposition(deadline=None):
    """Try to build a rank-40 decomposition using ALS for rank-10 per block."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 40))
    V = np.zeros((m * p, 40))
    W = np.zeros((p * n, 40))
    
    for block in range(4):
        if deadline and time.time() > deadline:
            return None
        # Try multiple seeds for rank-10
        found = None
        for seed in range(10):
            if deadline and time.time() > deadline:
                return None
            result = _als_block(10, block, max_iters=50, deadline=deadline, seed=seed)
            if result is not None:
                found = result
                break
        if found is None:
            return None
        Ub, Vb, Wb = found
        col = block * 10
        U[:, col:col + 10] = Ub
        V[:, col:col + 10] = Vb
        W[:, col:col + 10] = Wb
    
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
    deadline = start + 140

    # 1) Try rank-40 via ALS.
    try:
        result40 = _build_rank40_decomposition(deadline=deadline)
        if result40 is not None:
            U40, V40, W40 = result40
            if _verify(U40, V40, W40, tol=1e-6):
                rounded = _round_and_verify(U40, V40, W40)
                best_U, best_V, best_W = rounded if rounded else (U40, V40, W40)
                best_rank = 40
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
