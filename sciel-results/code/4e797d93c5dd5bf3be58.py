"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
DEADLINE = 100.0  # wall-clock seconds; safety margin under the 120s limit

# EVOLVE-BLOCK-START

def _tensor_positions(n, m, p):
    pos = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pos.append((i * m + j, j * p + k, k * n + i))
    return pos


def _make_tensor(n, m, p):
    Udim, Vdim, Wdim = n * m, m * p, p * n
    T = np.zeros((Udim, Vdim, Wdim), dtype=float)
    for (a, b, c) in _tensor_positions(n, m, p):
        T[a, b, c] = 1.0
    return T


def _verify(U, V, W, n, m, p):
    Udim, Vdim, Wdim = n * m, m * p, p * n
    rank = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    posset = set(_tensor_positions(n, m, p))
    for a in range(Udim):
        for b in range(Vdim):
            for d in range(Wdim):
                s = Fraction(0)
                for t in range(rank):
                    s += Uf[a][t] * Vf[b][t] * Wf[d][t]
                if s != (1 if (a, b, d) in posset else 0):
                    return False
    return True


def _float_residual(U, V, W, T):
    Trec = np.einsum('ic,jc,kc->ijk', U, V, W)
    R = Trec - T
    return float((R * R).sum())


def _snap_polish_verify(U, V, W, T, n, m, p, max_passes=80):
    Udim, Vdim, Wdim = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    Us = np.round(U * 2) / 2.0
    Vs = np.round(V * 2) / 2.0
    Ws = np.round(W * 2) / 2.0
    if _verify(Us.tolist(), Vs.tolist(), Ws.tolist(), n, m, p):
        return Us.tolist(), Vs.tolist(), Ws.tolist()
    # coordinate-descent polish on float residual, snapping coords to half-integers
    best = _float_residual(Us, Vs, Ws, T)
    for _ in range(max_passes):
        improved = False
        for mat, arr in ((Us, Us), (Vs, Vs), (Ws, Ws)):
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    x = arr[r, c]
                    base = round(x * 2) / 2.0
                    for cand in (base, base - 0.5, base + 0.5, base - 1.0, base + 1.0):
                        old = arr[r, c]
                        arr[r, c] = cand
                        val = _float_residual(Us, Vs, Ws, T)
                        if val < best - 1e-12:
                            best = val
                            improved = True
                        else:
                            arr[r, c] = old
        if best < 1e-12:
            break
        if not improved:
            break
    if best < 1e-9 and _verify(Us.tolist(), Vs.tolist(), Ws.tolist(), n, m, p):
        return Us.tolist(), Vs.tolist(), Ws.tolist()
    return None


def _search_tensor(T, dims, nmp, rank, time_budget, seed):
    Udim, Vdim, Wdim = dims
    n, m, p = nmp
    rng = np.random.default_rng(seed)
    total = Udim * rank + Vdim * rank + Wdim * rank
    start = time.time()

    def obj_and_grad(flat):
        U = flat[0:Udim * rank].reshape(Udim, rank)
        V = flat[Udim * rank:Udim * rank + Vdim * rank].reshape(Vdim, rank)
        W = flat[Udim * rank + Vdim * rank:].reshape(Wdim, rank)
        Trec = np.einsum('ic,jc,kc->ijk', U, V, W)
        R = Trec - T
        loss = float((R * R).sum())
        gU = 2.0 * np.einsum('ijk,jc,kc->ic', R, V, W)
        gV = 2.0 * np.einsum('ijk,ic,kc->jc', R, U, W)
        gW = 2.0 * np.einsum('ijk,ic,jc->kc', R, U, V)
        return loss, np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    for restart in range(400):
        if time.time() - start > time_budget:
            break
        scale = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
        flat0 = rng.standard_normal(total) * scale
        try:
            res = minimize(obj_and_grad, flat0, method='L-BFGS-B',
                           options={'maxiter': 4000, 'maxfun': 8000, 'gtol': 1e-14})
        except Exception:
            continue
        if res.fun < 1e-8:
            flat = res.x
            U = flat[0:Udim * rank].reshape(Udim, rank)
            V = flat[Udim * rank:Udim * rank + Vdim * rank].reshape(Vdim, rank)
            W = flat[Udim * rank + Vdim * rank:].reshape(Wdim, rank)
            sol = _snap_polish_verify(U, V, W, T, n, m, p)
            if sol is not None:
                return sol
    return None


def _build_236_from_233(U3, V3, W3):
    # U3 (6 x r3), V3 (9 x r3), W3 (6 x r3) -> [2,3,6] rank 2*r3
    n, m, p = 2, 3, 6
    r3 = len(U3[0])
    rank = 2 * r3
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c in range(r3):
        for d in range(2):
            col = c * 2 + d
            # U: A-dim of [1,1,2] is 1, both terms have a-coord 1
            for i in range(n * m):
                U[i][col] = U3[i][c]
            # V: index (j,k3) in V3 -> (j*3+k3)*2 + k' ; V'_0=[1,0], V'_1=[0,1]
            for j in range(3):
                for k3 in range(3):
                    vval = V3[j * 3 + k3][c]
                    V[(j * 3 + k3) * 2 + 0][col] = vval if d == 0 else 0.0
                    V[(j * 3 + k3) * 2 + 1][col] = vval if d == 1 else 0.0
            # W: index (k3,i) in W3 -> (k3*2+k')*2 + i ; W'_0=[1,0], W'_1=[0,1]
            for k3 in range(3):
                for i in range(2):
                    wval = W3[k3 * 2 + i][c]
                    W[(k3 * 2 + 0) * 2 + i][col] = wval if d == 0 else 0.0
                    W[(k3 * 2 + 1) * 2 + i][col] = wval if d == 1 else 0.0
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)
    best = (len(fallback["U"][0]), fallback)  # (rank, dict)

    def remaining():
        return max(0.0, DEADLINE - (time.time() - t0))

    # Stage 1: find [2,3,3] low rank, build [2,3,6] = [2,3,3] x [1,1,2]
    T3 = _make_tensor(2, 3, 3)
    for r3 in (7, 8, 9, 10, 12, 14):
        if remaining() < 8.0:
            break
        sol = _search_tensor(T3, (6, 9, 6), (2, 3, 3), r3, min(remaining() - 5.0, 25.0), seed=1000 + r3)
        if sol is not None:
            U, V, W = _build_236_from_233(*sol)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best[0]:
                    best = (r, {"U": U, "V": V, "W": W})
                break  # got a [2,3,3]-based solution; move to direct refinement

    # Stage 2: direct search on [2,3,6] to try to beat the current best
    T6 = _make_tensor(n, m, p)
    dims6 = (n * m, m * p, p * n)
    for r in (12, 13, best[0] - 1, 14, 15):
        if r <= 0 or r >= best[0]:
            continue
        if remaining() < 6.0:
            break
        sol = _search_tensor(T6, dims6, (n, m, p), r, min(remaining() - 4.0, 30.0), seed=5000 + r)
        if sol is not None:
            U, V, W = sol
            if _verify(U, V, W, n, m, p):
                best = (r, {"U": U, "V": V, "W": W})
                break

    return best[1]

# EVOLVE-BLOCK-END
