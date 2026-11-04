import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
N, M, P = 2, 3, 6
DEADLINE = 110.0  # safety margin under 120s


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _resid_u(u, v, w):
    """residual[i,j] = 1 - sum_c u_c[i,j]*v_c[j,*]... computed via w."""
    n, m, p = N, M, P
    # r[i,j] = 1 - sum_c w_c[i,*] * u_c[i,j] * v_c[j,*]
    # v_c[j,*] = sum_k v_c[j,k]  (constant in k)
    vs = np.zeros((m, p))
    # build vsum[j] = sum_k v[j,k]
    vsum = np.zeros(m)
    for j in range(m):
        s = 0.0
        for k in range(p):
            s += v[j * p + k]
        vsum[j] = s
    # wsum[i] = sum_k w[i,k]
    wsum = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(p):
            s += w[k * n + i]
        wsum[i] = s
    res = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for c in range(len(u)):
                s += u[c][i * m + j] * vsum[j] * wsum[i]
            res[i, j] = 1.0 - s
    return res


def _resid_v(u, v, w):
    n, m, p = N, M, P
    us = np.zeros(m)
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += u[i * m + j]
        us[j] = s
    wsum = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(p):
            s += w[k * n + i]
        wsum[i] = s
    res = np.zeros((m, p))
    for j in range(m):
        for k in range(p):
            s = 0.0
            for c in range(len(u)):
                s += us[j] * v[j * p + k] * wsum[k % n]
            # recompute properly below
    # redo cleanly
    res = np.zeros((m, p))
    for j in range(m):
        for k in range(p):
            s = 0.0
            for c in range(len(u)):
                s += us[j] * v[j * p + k] * wsum[k % n]
            res[j, k] = 1.0 - s
    return res


def _resid_w(u, v, w):
    n, m, p = N, M, P
    us = np.zeros(m)
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += u[i * m + j]
        us[j] = s
    vsum = np.zeros(p)
    for k in range(p):
        s = 0.0
        for j in range(m):
            s += v[j * p + k]
        vsum[k] = s
    res = np.zeros((p, n))
    for k in range(p):
        for i in range(n):
            s = 0.0
            for c in range(len(u)):
                s += us[0] * vsum[k] * w[k * n + i]
            # fix: us should be sum over i of u[i,j]; but u sum over i depends on j.
            # For w residual we need sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = 1
            # sum over j: sum_j u_c[i,j]*v_c[j,k] is (U^T V)_{i,k}
            pass
    # clean recompute
    res = np.zeros((p, n))
    for k in range(p):
        for i in range(n):
            s = 0.0
            for c in range(len(u)):
                uv = 0.0
                for j in range(m):
                    uv += u[c][i * m + j] * v[c][j * p + k]
                s += uv * w[c][k * n + i]
            res[k, i] = 1.0 - s
    return res


def _full_error(u, v, w):
    n, m, p = N, M, p
    n, m, p = N, M, P
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(u)):
                    s += u[c][i * m + j] * v[c][j * p + k] * w[c][k * n + i]
                err += (1.0 - s) ** 2
    return err


def _kronester_candidates():
    """Generate candidate rank-1 terms as Kronecker products of small support vectors."""
    n, m, p = N, M, P
    cands = []

    def all_vecs(size, max_support):
        out = []
        import itertools
        for support in range(1, max_support + 1):
            for idx in itertools.combinations(range(size), support):
                # each support entry = +/-1 or +/-1/2
                signs = list(itertools.product([1, -1, 0.5, -0.5], repeat=support))
                for sgn in signs:
                    vec = [0.0] * size
                    for a, b in zip(idx, sgn):
                        vec[a] = b
                    out.append(vec)
        return out

    un = all_vecs(n, n)
    um = all_vecs(m, m)
    up = all_vecs(p, p)

    # u[i,j] = un[i]*um[j]; v[j,k]=um[j]*up[k]; w[k,i]=up[k]*un[i]
    # consistency: u[i,j]*v[j,k]*w[k,i] = un[i]*um[j]*um[j]*up[k]*up[k]*un[i] = un[i]^2*um[j]^2*up[k]^2
    # For this to equal 1 (when all nonzero) we'd need un[i]=um[j]=up[k]=1. But with restricted
    # support we just need the product tensor to have entries in {0,1} matching a sub-rectangle.
    # Actually product = un[i]^2*um[j]^2*up[k]^2 which is 1 or 0.4 etc. We need it to be 0 or 1.
    # So restrict to entries in {0,1} for the product: require un,um,up entries in {0,1,-1}
    # so squares are 0 or 1.
    def all_vecs01(size):
        out = []
        import itertools
        for support in range(1, size + 1):
            for idx in itertools.combinations(range(size), support):
                for sgn in itertools.product([1, -1], repeat=support):
                    vec = [0.0] * size
                    for a, b in zip(idx, sgn):
                        vec[a] = b
                    out.append(vec)
        return out

    un = all_vecs01(n)
    um = all_vecs01(m)
    up = all_vecs01(p)

    for a in un:
        for b in um:
            for c in up:
                u = [a[i] * b[j] for i in range(n) for j in range(m)]
                v = [b[j] * c[k] for j in range(m) for k in range(p)]
                w = [c[k] * a[i] for k in range(p) for i in range(n)]
                cands.append((u, v, w))
    return cands


def _search(deadline):
    n, m, p = N, M, P
    cands = _kronester_candidates()
    # precompute each candidate's contribution pattern as (u, v, w) lists
    rng = np.random.default_rng(12345)
    best = None

    def eval_combo(sel):
        u = [cands[i][0] for i in sel]
        v = [cands[i][1] for i in sel]
        w = [cands[i][2] for i in sel]
        return _full_error(u, v, w)

    # greedy
    sel = []
    cur_err = _full_error([], [], [])  # all ones residual = n*m*p
    used = set()
    while time.time() < deadline:
        best_idx = None
        best_err = cur_err
        for ci in range(len(cands)):
            if ci in used:
                continue
            sel2 = sel + [ci]
            e = eval_combo(sel2)
            if e < best_err - 1e-12:
                best_err = e
                best_idx = ci
        if best_idx is None:
            break
        sel.append(best_idx)
        used.add(best_idx)
        cur_err = best_err
        if cur_err < 1e-9:
            break

    # local search: try swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(len(sel)):
            for ci in range(len(cands)):
                if ci in used and ci != sel[pos]:
                    continue
                if ci == sel[pos]:
                    continue
                sel2 = sel[:pos] + [ci] + sel[pos + 1:]
                used2 = (used - {sel[pos]}) | {ci}
                e = eval_combo(sel2)
                if e < cur_err - 1e-12:
                    sel = sel2
                    used = used2
                    cur_err = e
                    improved = True
        if cur_err < 1e-9:
            break

    # build result
    u = [cands[i][0] for i in sel]
    v = [cands[i][1] for i in sel]
    w = [cands[i][2] for i in sel]
    if cur_err < 1e-9:
        rank = len(sel)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for c in range(rank):
            for r in range(n * m):
                U[r][c] = float(u[c][r])
            for r in range(m * p):
                V[r][c] = float(v[c][r])
            for r in range(p * n):
                W[r][c] = float(w[c][r])
        return {"U": U, "V": V, "W": W, "_rank": rank}
    return None


def _verify(res):
    U, V, W = res["U"], res["V"], res["W"]
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Try search
    res = None
    try:
        res = _search(deadline)
    except Exception:
        res = None

    if res is not None and _verify(res):
        rank = len(res["U"][0])
        if rank < N * M * P:
            return res

    # Fallback schoolbook
    return _schoolbook()
