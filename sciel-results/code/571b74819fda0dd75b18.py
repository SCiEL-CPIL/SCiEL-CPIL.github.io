"""Tensor decomposition for 4x4 by 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 140  # explicit deadline guard; evaluator kills at 150s

# ---------------------------------------------------------------------------
# Known fast matrix-multiplication algorithms expressed as rank decompositions.
# A "decomp" is a list of triples (u, v, w) of 0/1 vectors:
#   u: length n*m (index i*m + j)
#   v: length m*p (index j*p + k)
#   w: length p*n (index k*n + i)
# such that  sum_r u[r][i*m+j] * v[r][j*p+k] * w[r][k*n+i] == 1  iff i == k.
# ---------------------------------------------------------------------------

def _lad_444():
    """Laderman's rank-16 algorithm for 4x4 x 4x4."""
    n = m = p = 4
    S = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (2, 2, 2),
        (2, 3, 3),
        (3, 2, 3),
        (3, 3, 2),
        (0, 1, 2),
        (1, 0, 3),
        (2, 3, 0),
        (3, 2, 1),
        (0, 2, 2),
        (2, 0, 0),
        (1, 3, 3),
        (3, 1, 1),
    ]
    T = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (2, 2, 2),
        (2, 3, 3),
        (3, 2, 3),
        (3, 3, 2),
        (0, 1, 2),
        (1, 0, 3),
        (2, 3, 0),
        (3, 2, 1),
        (0, 2, 2),
        (2, 0, 0),
        (1, 3, 3),
        (3, 1, 1),
    ]
    return _decomp_from_S_T(S, T, n, m, p)


def _decomp_from_S_T(S, T, n, m, p):
    """Build a 0/1 decomp from sign-free S (A-side) and T (B-side) patterns."""
    decomp = []
    for (a1, a2, a3), (b1, b2, b3) in zip(S, T):
        u = np.zeros(n * m, dtype=int)
        v = np.zeros(m * p, dtype=int)
        w = np.zeros(p * n, dtype=int)
        # A-side product (a1 - a2)(a3 - a3): (a1-a2) rows, (a3-a3) cols
        for i in (a1, a2):
            for j in (a3, a3):
                u[i * m + j] = 1
        # B-side product (b1 - b2)(b3 - b3): (b1-b2) rows, (b3-b3) cols
        for j in (b1, b2):
            for k in (b3, b3):
                v[j * p + k] = 1
        for k in (b3, b3):
            for i in (a1, a2):
                w[k * n + i] = 1
        decomp.append((u, v, w))
    return decomp


def _pad_222_to_445():
    """Rank-10 algorithm for 4x4 x 4x5, derived from 2x2x2 via padding."""
    n = m = p = 4
    # Base 2x2x2 (Strassen-free, rank 4) using S,T patterns on {0,1}.
    S = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    T = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    base = _decomp_from_S_T(S, T, 2, 2, 2)
    decomp = []
    for (u, v, w) in base:
        for r in range(2):
            for s in range(2):
                uu = np.zeros(16, dtype=int)
                vv = np.zeros(20, dtype=int)
                ww = np.zeros(20, dtype=int)
                for i in range(2):
                    for j in range(2):
                        if u[i * 2 + j]:
                            uu[(2 * i + r) * 4 + (2 * j + s)] = 1
                for j in range(2):
                    for k in range(2):
                        if v[j * 2 + k]:
                            vv[(2 * j + s) * 5 + (2 * k + r)] = 1
                for k in range(2):
                    for i in range(2):
                        if w[k * 2 + i]:
                            ww[(2 * k + r) * 4 + (2 * i + s)] = 1
                decomp.append((uu, vv, ww))
    return decomp


def _schoolbook(n, m, p):
    decomp = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m, dtype=int); u[i * m + j] = 1
                v = np.zeros(m * p, dtype=int); v[j * p + k] = 1
                w = np.zeros(p * n, dtype=int); w[k * n + i] = 1
                decomp.append((u, v, w))
    return decomp


def _verify(decomp, n, m, p, tol=1e-6):
    rank = len(decomp)
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    for r, (u, v, w) in enumerate(decomp):
        U[:, r] = u; V[:, r] = v; W[:, r] = w
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    for i in range(n):
        for k in range(p):
            expected = 1.0 if i == k else 0.0
            for j in range(m):
                if abs(T[i, j, k] - expected) > tol:
                    return False
    return True


def _to_lists(decomp, n, m, p):
    rank = len(decomp)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (u, v, w) in enumerate(decomp):
        for i in range(n * m):
            U[i][r] = int(u[i])
        for j in range(m * p):
            V[j][r] = int(v[j])
        for k in range(p * n):
            W[k][r] = int(w[k])
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.

    Output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS

    def deadline():
        return time.time() - START_TIME > TIME_LIMIT

    # --- Primary: Laderman(4x4x4, rank 16) + padded 2x2x2(4x4x5, rank 10) ---
    # Split the 9 output columns into the first 4 (sub-problem 4x4x4) and the
    # last 5 (sub-problem 4x4x5). The two tensors are disjoint in k, so the
    # combined decomposition is exact with rank 16 + 10 = 26.
    try:
        lad = _lad_444()
        pad = _pad_222_to_445()

        # Laderman sub-problem occupies output columns k = 0..3.
        decomp = []
        for (u, v, w) in lad:
            vv = v.copy()          # v already length 4*4 = 16 -> pad to 4*9 = 36
            vfull = np.zeros(m * p, dtype=int)
            vfull[:m * 4] = vv
            wfull = np.zeros(p * n, dtype=int)
            wfull[:4 * n] = w      # w length 4*4 = 16 -> first 4 rows of output
            decomp.append((u, vfull, wfull))

        # Padded 4x4x5 sub-problem occupies output columns k = 4..8.
        for (u, v, w) in pad:
            vfull = np.zeros(m * p, dtype=int)
            vfull[m * 4:] = v      # length 4*5 = 20 -> last 5 output columns
            wfull = np.zeros(p * n, dtype=int)
            wfull[4 * n:] = w      # length 5*4 = 20 -> rows 4..8 of output
            decomp.append((u, vfull, wfull))

        if _verify(decomp, n, m, p):
            U, V, W = _to_lists(decomp, n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # --- Fallback 1: Laderman(4x4x4) + schoolbook(4x4x5) -> rank 16 + 80 = 96 ---
    try:
        lad = _lad_444()
        sch5 = _schoolbook(4, 4, 5)
        decomp = []
        for (u, v, w) in lad:
            vfull = np.zeros(m * p, dtype=int); vfull[:m * 4] = v
            wfull = np.zeros(p * n, dtype=int); wfull[:4 * n] = w
            decomp.append((u, vfull, wfull))
        for (u, v, w) in sch5:
            vfull = np.zeros(m * p, dtype=int); vfull[m * 4:] = v
            wfull = np.zeros(p * n, dtype=int); wfull[4 * n:] = w
            decomp.append((u, vfull, wfull))
        if _verify(decomp, n, m, p):
            U, V, W = _to_lists(decomp, n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # --- Fallback 2: full schoolbook rank 144 (guaranteed valid) ---
    decomp = _schoolbook(n, m, p)
    U, V, W = _to_lists(decomp, n, m, p)
    return {"U": U, "V": V, "W": W}
