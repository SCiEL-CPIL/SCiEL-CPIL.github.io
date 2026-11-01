"""Tensor rank minimization for 2x3 * 3x9 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _triples(n, m, p):
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def _residuals(x, U_idx, V_idx, W_idx, r, n, m, p, triples):
    """x is flat params; compute vector of (sum - 1) per triple."""
    U = x[0:U_idx].reshape(n * m, r)
    V = x[U_idx:V_idx].reshape(m * p, r)
    W = x[V_idx:W_idx].reshape(p * n, r)
    out = np.empty(len(triples))
    for t, (i, j, k) in enumerate(triples):
        out[t] = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i]) - 1.0
    return out

def construct():
    n, m, p = DIMENSIONS
    triples = _triples(n, m, p)
    rng = random.Random(7)
    deadline = time.time() + 105

    # Schoolbook baseline (rank 54), always valid
    rank0 = n * m * p
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = rank0

    def snap(L):
        for row in L:
            for idx, v in enumerate(row):
                if abs(v - round(v)) < 1e-6:
                    row[idx] = float(round(v))
        return L

    # Search candidate ranks below 54 using least_squares
    lo = max(n * m, m * p, p * n)  # 27
    for r in range(lo, best_rank):
        if time.time() > deadline - 15:
            break
        per = max(0.4, min(2.5, (deadline - time.time()) / max(1, best_rank - r)))
        U_idx = n * m * r
        V_idx = U_idx + m * p * r
        W_idx = V_idx + p * n * r
        found = False
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            x0 = np.random.default_rng(rng.randrange(1 << 30)).normal(size=(W_idx,))
            try:
                from scipy.optimize import least_squares
                res = least_squares(
                    _residuals, x0, args=(U_idx, V_idx, W_idx, r, n, m, p, triples),
                    method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
                if res.cost < 1e-14:
                    x = res.x
                    U = x[0:U_idx].reshape(n * m, r)
                    V = x[U_idx:V_idx].reshape(m * p, r)
                    W = x[V_idx:W_idx].reshape(p * n, r)
                    # verify exactly
                    ok = True
                    for (i, j, k) in triples:
                        s = float(np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i]))
                        if abs(s - 1.0) > 1e-7:
                            ok = False
                            break
                    if ok:
                        best = {"U": snap(U.tolist()), "V": snap(V.tolist()), "W": snap(W.tolist())}
                        best_rank = r
                        found = True
                        break
            except Exception:
                pass
        if found:
            break
    return best

# EVOLVE-BLOCK-END
