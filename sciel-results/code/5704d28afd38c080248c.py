"""Matrix multiplication 4x5x6 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-7):
    r = U.shape[1]
    for i in range(n):
        Ui = U[i * m:(i + 1) * m, :]
        for k in range(p):
            Wk = W[k * n:(k + 1) * n, :]
            for j in range(m):
                s = np.einsum('r,r,r->', Ui[j], V[j * p + k], Wk[i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank1_residual(u, v, w, n, m, p):
    """residual[i,j,k] = u[i,j]*v[j,k]*w[k,i]."""
    res = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        ui = u[i]
        if not np.any(ui):
            continue
        for k in range(p):
            wk = w[k, i]
            if wk == 0.0:
                continue
            for j in range(m):
                res[i, j, k] = ui[j] * v[j, k] * wk
    return res


def _greedy_sparse(n, m, p, max_rank, deadline):
    """Greedy sparse rank-r factorization. Each term is a rank-1 tensor u⊗v⊗w.
    u[i,j]=1 iff A[i,j] feeds C[i,k]; v[j,k]=1 iff B[j,k] feeds C[i,k];
    w[k,i]=1 iff C[i,k] is produced. Terms with zero residual contribution
    are dropped (cancellation of redundant terms)."""
    target = np.ones((n, m, p), dtype=float)
    residual = target.copy()
    U = []
    V = []
    W = []
    for _ in range(max_rank):
        if time.time() > deadline:
            break
        best = None
        for i in range(n):
            for k in range(p):
                u = np.zeros((n, m)); u[i, :] = 1.0
                w = np.zeros((p, n)); w[k, i] = 1.0
                # residual weighted by this (i,k): sum_j u[i,j]*v[j,k]*w[k,i]
                # = v[j,k] for j; we want max_j residual[i,j,k]
                v = np.zeros((m, p))
                best_j = np.argmax(residual[i, :, k])
                v[best_j, k] = 1.0
                if best is None or residual[i, best_j, k] > best[0]:
                    best = (residual[i, best_j, k], i, k, best_j)
        if best is None or best[0] <= 1e-9:
            break
        _, i, k, j = best
        u = np.zeros((n, m)); u[i, :] = 1.0
        v = np.zeros((m, p)); v[j, k] = 1.0
        w = np.zeros((p, n)); w[k, i] = 1.0
        U.append(u); V.append(v); W.append(w)
        residual = target - sum(_rank1_residual(a, b, c, n, m, p) for a, b, c in zip(U, V, W))
    if np.max(np.abs(residual)) > 1e-7:
        return None
    r = len(U)
    Um = np.zeros((n * m, r)); Vm = np.zeros((m * p, r)); Wm = np.zeros((p * n, r))
    for c in range(r):
        Um[U[c].nonzero()] = 1.0
        Vm[V[c].nonzero()] = 1.0
        Wm[W[c].nonzero()] = 1.0
    return Um, Vm, Wm


def _try_rank(n, m, p, rank, deadline):
    """Solve the bilinear system for rank terms with exact rational arithmetic,
    then pad to the full interface indices."""
    nmp = n * m * p
    # Build constraint matrix A: row (i,j,k), col (r,a,b) where a in [0,n), b in [0,p)
    # A[row, (r*n+p)*a + b] = 1  iff  b == k  AND  (a == i  OR  a == i-1)
    A = np.zeros((nmp, rank * n * p), dtype=float)
    rhs = np.ones(nmp)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m * p + j * p + k
                for r in range(rank):
                    base = (r * n + k) * p
                    A[row, base] = 1.0          # a == i
                    if i - 1 >= 0:
                        A[row, base + (i - 1)] = 1.0
    # Solve least squares
    try:
        x, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    except Exception:
        return None
    # Verify in float
    pred = A @ x
    if np.max(np.abs(pred - rhs)) > 1e-6:
        return None
    # Refine with exact rational arithmetic (round to nearest integer)
    xr = np.rint(x)
    if np.max(np.abs(A @ xr - rhs)) > 1e-9:
        return None
    # Build U, V, W (padded)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r in range(rank):
        for a in range(n):
            for b in range(p):
            # x index: (r*n+p)*a + b
                val = xr[(r * n + a) * p + b]
                if val == 0:
                    continue
                for i in (a, a - 1):
                    if 0 <= i < n:
                        U[i * m + a, r] += val
                for j in range(m):
                    V[j * p + b, r] += val
                W[b * n + a, r] += val
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)


def construct():
    deadline = time.time() + 125  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    best_rank = n * m * p  # schoolbook rank
    best_U, best_V, best_W = _schoolbook(n, m, p)

    # Greedy sparse search: often finds a valid (possibly lower) rank quickly.
    gres = _greedy_sparse(n, m, p, best_rank, deadline)
    if gres is not None:
        U, V, W = gres
        if _verify(U, V, W, n, m, p):
            best_rank = U.shape[1]
            best_U, best_V, best_W = U, V, W

    # Rank-reduction search: try to find a valid decomposition with fewer terms.
    for rank in range(max(30, best_rank - 40), best_rank + 1):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
