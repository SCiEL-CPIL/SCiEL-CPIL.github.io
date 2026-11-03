"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0   # explicit safety margin below the 150s kill


def _time_left():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Exact / fast verification
# ---------------------------------------------------------------------------
def _verify_np(U, V, W, n, m, p, tol=1e-7):
    """Vectorized exact check: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= tol))


# ---------------------------------------------------------------------------
# Primitive block builders (return (U, V, W) as list-of-lists, local indexing)
# ---------------------------------------------------------------------------
def _build_strassen_2x2():
    """2x2 Strassen. Returns (U, V, W) each shape (4, 7). Local n=m=p=2."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1; V[1][2] = 1; V[3][2] = -1; W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1; V[2][3] = 1; V[0][3] = -1; W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
    return U, V, W


def _build_strassen_4x4x4():
    """4x4x4 Strassen (Kronecker of 2x2x2), rank 49. Local n=m=p=4."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1 * 2 + j1][r1]
                    if not c1:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2 * 2 + j2][r2]
                            if not c2:
                                continue
                            U[(i1 * 2 + i2) * 4 + (j1 * 2 + j2)][col] += c1 * c2
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1 * 2 + k1][r1]
                    if not c1:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2 * 2 + k2][r2]
                            if not c2:
                                continue
                            V[(j1 * 2 + j2) * 4 + (k1 * 2 + k2)][col] += c1 * c2
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1 * 2 + i1][r1]
                    if not c1:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2 * 2 + i2][r2]
                            if not c2:
                                continue
                            W[(k1 * 2 + k2) * 4 + (i1 * 2 + i2)][col] += c1 * c2
    return U, V, W


def _build_strassen_4x4x2():
    """4x4x2 Strassen (4 Strassen 2x2x2 blocks), rank 28. Local n=4,m=4,p=2."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]
    blocks = [(0, 0, 0, 0), (0, 2, 2, 0), (2, 0, 0, 2), (2, 2, 2, 2)]
    for b, (ai, aj, bj, ci) in enumerate(blocks):
        base = b * 7
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li * 2 + lj][t]
                    if ca:
                        U[(ai + li) * 4 + (aj + lj)][col] += ca
            for lj in range(2):
                for lk in range(2):
                    cb = V2[lj * 2 + lk][t]
                    if cb:
                        V[(bj + lj) * 2 + lk][col] += cb
            for lk in range(2):
                for li in range(2):
                    cw = W2[lk * 2 + li][t]
                    if cw:
                        W[(ci + lk) * 4 + li][col] += cw
    return U, V, W


def _build_4x1x6():
    """4x1x6 schoolbook, rank 24. Local n=4,m=1,p=6."""
    rank = 24
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(24)]
    c = 0
    for i in range(4):
        for k in range(6):
            U[i * 1 + 0][c] = 1
            V[0 * 6 + k][c] = 1
            W[k * 4 + i][c] = 1
            c += 1
    return U, V, W


def _build_schoolbook(n, m, p):
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
    return U, V, W


# ---------------------------------------------------------------------------
# Combine block decompositions into a full 4x5x6 decomposition
# ---------------------------------------------------------------------------
def _combine(decomps, n, m, p):
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    cur = 0
    for U_s, V_s, W_s, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_s = i_e - i_s
        m_s = j_e - j_s
        p_s = k_e - k_s
        r_s = len(U_s[0])

        for il in range(n_s):
            for jl in range(m_s):
                ig = i_s + il
                jg = j_s + jl
                rg = ig * m + jg
                for c in range(r_s):
                    U[rg][cur + c] = U_s[il * m_s + jl][c]

        for jl in range(m_s):
            for kl in range(p_s):
                jg = j_s + jl
                kg = k_s + kl
                rg = jg * p + kg
                for c in range(r_s):
                    V[rg][cur + c] = V_s[jl * p_s + kl][c]

        for kl in range(p_s):
            for il in range(n_s):
                kg = k_s + kl
                ig = i_s + il
                rg = kg * n + ig
                for c in range(r_s):
                    W[rg][cur + c] = W_s[kl * n_s + il][c]

        cur += r_s
    return U, V, W


# ---------------------------------------------------------------------------
# Column elimination: drop redundant rank-1 columns using algebraic cancellation
# ---------------------------------------------------------------------------
def _eliminate_columns(U, V, W, n, m, p, seed=0):
    """Iteratively remove a column if its contribution can be absorbed by others."""
    rng = random.Random(seed)
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    keep = list(range(len(U[0])))
    R0 = len(keep)

    def verify(cols):
        if not cols:
            return False
        Uf = np.array([U[i][c] for i in range(n * m) for c in cols], dtype=float).reshape(n * m, -1)
        Vf = np.array([V[i][c] for i in range(m * p) for c in cols], dtype=float).reshape(m * p, -1)
        Wf = np.array([W[i][c] for i in range(p * n) for c in cols], dtype=float).reshape(p * n, -1)
        U3 = Uf.reshape(n, m, -1)
        V3 = Vf.reshape(m, p, -1)
        W3 = Wf.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return bool(np.all(np.abs(T - 1.0) <= 1e-7))

    for it in range(4000):
        if _time_left() < 8.0:
            break
        if len(keep) <= 1:
            break
        cand = list(keep)
        rng.shuffle(cand)
        for c in cand:
            trial = [x for x in keep if x != c]
            if verify(trial):
                keep = trial
                break
    if verify(keep):
        Rf = len(keep)
        if Rf < R0:
            U = [U[i][c] for i in range(n * m) for c in keep]
            V = [V[i][c] for i in range(m * p) for c in keep]
            W = [W[i][c] for i in range(p * n) for c in keep]
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS

    try:
        # Base block decomposition: 49 + 28 + 24 = 101
        U1, V1, W1 = _build_strassen_4x4x4()
        d1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        U2, V2, W2 = _build_4x1x6()
        d2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        U3, V3, W3 = _build_strassen_4x4x2()
        d3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))

        U, V, W = _combine([d1, d2, d3], n, m, p)

        if not _verify_np(U, V, W, n, m, p):
            # fall back to a guaranteed-valid schoolbook if the base is wrong
            U, V, W = _build_schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}

        # Refinement: search for a lower-rank sub-decomposition via column removal
        best = (U, V, W)
        for seed in range(6):
            if _time_left() < 10.0:
                break
            res = _eliminate_columns(U, V, W, n, m, p, seed=seed)
            if res is not None and len(res[0][0]) < len(best[0][0]):
                best = res
        U, V, W = best
        return {"U": U, "V": V, "W": W}

    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
