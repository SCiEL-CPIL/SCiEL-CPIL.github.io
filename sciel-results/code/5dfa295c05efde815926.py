"""Matrix multiplication 2x4x7 - rank-minimal decomposition via randomized search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
DEADLINE = 145.0  # safety margin under the 150s evaluator kill

def _time_left():
    return DEADLINE - time.time()

def _strassen_solution():
    """Guaranteed rank-50 decomposition (Strassen blocks + direct tail)."""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
        col += 1
        U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
        col += 1
        U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    add_strassen(0, 0, 0); add_strassen(0, 0, 2)
    add_strassen(0, 2, 0); add_strassen(0, 2, 2)
    add_strassen(0, 0, 4); add_strassen(0, 2, 4)
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _merge_rank1(U, V, W):
    """Merge any two rank-1 terms whose U and V vectors are proportional."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    used = [False] * r
    newU, newV, newW = [], [], []
    for a in range(r):
        if used[a]:
            continue
        ua, va, wa = U[:, a], V[:, a], W[:, a]
        for b in range(a + 1, r):
            if used[b]:
                continue
            ub, vb = U[:, b], V[:, b]
            na, nb = np.linalg.norm(ua), np.linalg.norm(ub)
            if na < 1e-12 or nb < 1e-12:
                continue
            cos = abs(np.dot(ua, ub)) / (na * nb)
            if cos < 0.999999:
                continue
            mva = np.linalg.norm(va); mvb = np.linalg.norm(vb)
            if mva < 1e-12 or mvb < 1e-12:
                continue
            cosv = abs(np.dot(va, vb)) / (mva * mvb)
            if cosv < 0.999999:
                continue
            used[b] = True
            wa = wa + np.dot(ua, ub) / (na * na) * W[:, b]
            newU.append(ua); newV.append(va); newW.append(wa)
            break
        else:
            newU.append(ua); newV.append(va); newW.append(wa)
    U = np.column_stack(newU); V = np.column_stack(newV); W = np.column_stack(newW)
    return U, V, W

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    T2 = T.reshape(n*m, m*p, p*n)
    return bool(np.allclose(T2, _build_tensor(), atol=1e-8))

def _find_decomposition(T, rank, rng):
    """Randomized: pick random U,V, solve for W, verify."""
    n, m, p = DIMENSIONS
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    U, V = _merge_rank1(U, V, np.zeros((p*n, rank)))
    if U.shape[1] < rank:
        return None
    T2 = T.reshape(n*m, m*p, p*n)
    T2 = np.moveaxis(T2, 2, 0)  # (p*n, n*m, m*p)
    rhs = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
    rhs = np.moveaxis(rhs, 2, 0)         # (rank, n*m, m*p)
    A = rhs.reshape(rank, -1)
    b = T2.reshape(-1)
    try:
        Wflat, res, rankA, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    W = Wflat.reshape(rank, p*n)
    if not _verify(U, V, W):
        return None
    U, V, W = _merge_rank1(U, V, W)
    return U, V, W

def _to_lists(U, V, W):
    U = np.array(U); V = np.array(V); W = np.array(W)
    r = U.shape[1]
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }

def construct():
    n, m, p = DIMENSIONS
    random.seed(12345)
    rng = np.random.default_rng(12345)
    T = _build_tensor()

    # Guaranteed fallback.
    best = _strassen_solution()
    best_rank = 50

    # Rank-1 merge of the fallback (free improvement if any).
    Uf, Vf, Wf = _merge_rank1(best["U"], best["V"], best["W"])
    if _verify(Uf, Vf, Wf) and Uf.shape[1] < best_rank:
        best_rank = Uf.shape[1]
        best = _to_lists(Uf, Vf, Wf)

    # Randomized search for lower ranks.
    for rank in range(best_rank - 1, 30, -1):
        if _time_left() < 8.0:
            break
        # Number of attempts scaled by remaining time.
        attempts = max(1, int(_time_left() // 2.5))
        for _ in range(attempts):
            if _time_left() < 3.0:
                break
            res = _find_decomposition(T, rank, rng)
            if res is not None:
                U, V, W = res
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best = _to_lists(U, V, W)
                break

    if _time_left() < 0.5:
        return best
    return best
# EVOLVE-BLOCK-END
