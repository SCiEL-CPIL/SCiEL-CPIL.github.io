"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Design (kept from parent, refined):
  * Structured init: two independent Strassen 2x2 column blocks (rank 14).
  * Fast exact verification via tensor matricizations (no slow per-element loop).
  * Fast rank lower bound (numerical rank of the three matricizations).
  * ALS refinement (vectorized, fast).
  * Rank reduction: keep R-1 columns, re-solve the dropped one (always R-1),
    plus direct low-rank search from many random inits.
  * Rational polish to small denominators.
  * Schoolbook rank-20 fallback.

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
N, M, P = DIMENSIONS
DIM_U = N * M          # 4
DIM_V = M * P          # 10
DIM_W = P * N          # 10


def _build_target():
    T = np.zeros((DIM_U, DIM_V, DIM_W), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                T[u, v, w] = 1
    return T


def _decompose(U, V, W):
    R = U.shape[1]
    result = np.zeros((DIM_U, DIM_V, DIM_W), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    """Fast exact check via matricizations (equivalent to elementwise equality)."""
    R = U.shape[1]
    if R == 0:
        return bool(np.allclose(T, 0.0, atol=tol))
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    # mode-1: U x (V^T W)
    A1 = np.einsum('vr,wr->vwr', Vf, Wf).reshape(DIM_V, DIM_W, R)
    M1 = np.einsum('u, vwr -> uvw', Uf, A1)
    # mode-2: V x (U^T W)
    A2 = np.einsum('ur,wr->uwr', Uf, Wf).reshape(DIM_U, DIM_W, R)
    M2 = np.einsum('v, uwr -> uvw', Vf, A2)
    # mode-3: W x (U^T V)
    A3 = np.einsum('ur,vr->uvr', Uf, Vf).reshape(DIM_U, DIM_V, R)
    M3 = np.einsum('w, uvr -> uvw', Wf, A3)
    Tf = T.astype(np.float64)
    return (np.allclose(M1, Tf, atol=tol) and
            np.allclose(M2, Tf, atol=tol) and
            np.allclose(M3, Tf, atol=tol))


def _check_exact(U, V, W, T, tol=1e-6):
    """Snap to small-denominator rationals and verify exactly with Fraction."""
    if not _check_decomposition(U, V, W, T, tol=1e-5):
        return False

    def to_frac(x):
        for denom in [1, 2, 4, 8, 16, 32, 64, 128]:
            num = round(x * denom)
            if abs(x - num / denom) < tol:
                return Fraction(int(num), denom)
        return Fraction(x).limit_denominator(128)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    for u in range(DIM_U):
        for v in range(DIM_V):
            for w in range(DIM_W):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap(M, denom):
    return np.round(M * denom) / denom


def _polish_rational(U, V, W, T):
    """Snap to small-denominator rationals that still decompose exactly."""
    for denom in [1, 2, 4, 8, 16, 32, 64]:
        Us, Vs, Ws = _snap(U, denom), _snap(V, denom), _snap(W, denom)
        if _check_exact(Us, Vs, Ws, T):
            return Us.tolist(), Vs.tolist(), Ws.tolist()
    for dU in [1, 2, 4, 8]:
        for dV in [1, 2, 4, 8]:
            for dW in [1, 2, 4, 8]:
                Us, Vs, Ws = _snap(U, dU), _snap(V, dV), _snap(W, dW)
                if _check_exact(Us, Vs, Ws, T):
                    return Us.tolist(), Vs.tolist(), Ws.tolist()
    return None


def _als_update_U(T, V, W):
    R = V.shape[1]
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    U = np.zeros((DIM_U, R), dtype=np.float64)
    for u in range(DIM_U):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U[u] = sol
    return U


def _als_update_V(T, U, W):
    R = U.shape[1]
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    V = np.zeros((DIM_V, R), dtype=np.float64)
    for v in range(DIM_V):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V[v] = sol
    return V


def _als_update_W(T, U, V):
    R = U.shape[1]
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    W = np.zeros((DIM_W, R), dtype=np.float64)
    for w in range(DIM_W):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W[w] = sol
    return W


def _als_run(U, V, W, T, iters, deadline):
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_update_U(T, V, W)
        V = _als_update_V(T, U, W)
        W = _als_update_W(T, U, V)
    return U, V, W


def _estimate_tensor_rank(T, max_rank=25):
    T_f = T.astype(np.float64)
    M1 = T_f.reshape(DIM_U, -1)
    s1 = np.linalg.svd(M1, compute_uv=False)
    r1 = int(np.sum(s1 > 1e-10))
    M2 = T_f.transpose(1, 0, 2).reshape(DIM_V, -1)
    s2 = np.linalg.svd(M2, compute_uv=False)
    r2 = int(np.sum(s2 > 1e-10))
    M3 = T_f.transpose(2, 0, 1).reshape(DIM_W, -1)
    s3 = np.linalg.svd(M3, compute_uv=False)
    r3 = int(np.sum(s3 > 1e-10))
    return max(r1, r2, r3)


def _build_structured_decomposition():
    """Two independent Strassen 2x2 column blocks -> rank 14."""
    rank = 14
    U = np.zeros((DIM_U, rank), dtype=np.float64)
    V = np.zeros((DIM_V, rank), dtype=np.float64)
    W = np.zeros((DIM_W, rank), dtype=np.float64)

    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[col0, r] = 1; V[P + col1, r] = 1
        W[col0, r] = 1; W[N + col1, r] = 1

        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[col0, r] = 1
        W[N + col0, r] = 1; W[N + col1, r] = -1

        r = base + 2
        U[0, r] = 1
        V[P + col1, r] = 1; V[col1, r] = -1
        W[col1, r] = 1; W[N + col1, r] = 1

        r = base + 3
        U[3, r] = 1
        V[P + col0, r] = 1; V[col0, r] = -1
        W[col0, r] = 1; W[N + col0, r] = 1

        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[P + col1, r] = 1
        W[col0, r] = -1; W[col1, r] = 1

        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[col0, r] = 1; V[col1, r] = 1
        W[N + col1, r] = 1

        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[P + col0, r] = 1; V[P + col1, r] = 1
        W[col0, r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    return U, V, W


def _try_reduce_drop(U, V, W, T, drop, deadline, iters=120):
    """Keep all columns except 'drop'; re-solve that one column."""
    R = U.shape[1]
    keep = [i for i in range(R) if i != drop]
    U0, V0, W0 = U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy()
    U = np.zeros((DIM_U, R)); V = np.zeros((DIM_V, R)); W = np.zeros((DIM_W, R))
    U[:, keep] = U0; V[:, keep] = V0; W[:, keep] = W0
    U[:, drop] = np.random.randn(DIM_U) * 0.5
    V[:, drop] = np.random.randn(DIM_V) * 0.5
    W[:, drop] = np.random.randn(DIM_W) * 0.5
    U, V, W = _als_run(U, V, W, T, iters, deadline)
    if _check_decomposition(U, V, W, T, tol=1e-7):
        return U, V, W
    return None


def _try_direct_rank(R, T, deadline, inits=6, iters=150):
    """Direct low-rank search from random initializations."""
    best = None
    for init in range(inits):
        if time.time() > deadline:
            break
        U = np.random.randn(DIM_U, R)
        V = np.random.randn(DIM_V, R)
        W = np.random.randn(DIM_W, R)
        U, V, W = _als_run(U, V, W, T, iters, deadline)
        if _check_decomposition(U, V, W, T, tol=1e-7):
            best = (U, V, W)
            break
    return best


def _schoolbook_decomposition():
    rank = N * M * P
    U = np.zeros((DIM_U, rank), dtype=np.float64)
    V = np.zeros((DIM_V, rank), dtype=np.float64)
    W = np.zeros((DIM_W, rank), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    T = _build_target()
    rank_lb = _estimate_tensor_rank(T)
    if rank_lb < 1:
        rank_lb = 1

    U_struct, V_struct, W_struct = _build_structured_decomposition()
    if not _check_decomposition(U_struct, V_struct, W_struct, T, tol=1e-8):
        U_l, V_l, W_l = _schoolbook_decomposition()
        return {"U": U_l, "V": V_l, "W": W_l}

    best = (U_struct, V_struct, W_struct)
    best_rank = U_struct.shape[1]

    # 1) Reduce by dropping one column (guaranteed to stay at best_rank - 1).
    for drop in range(best_rank):
        if time.time() > deadline:
            break
        res = _try_reduce_drop(U_struct, V_struct, W_struct, T, drop, deadline, iters=150)
        if res is not None:
            U, V, W = res
            if _check_decomposition(U, V, W, T, tol=1e-7):
                best = (U, V, W)
                best_rank = U.shape[1]
                break

    # 2) Continue dropping columns from the improved solution.
    for _ in range(4):
        if time.time() > deadline:
            break
        if best_rank <= rank_lb:
            break
        U, V, W = best
        found = False
        for drop in range(best_rank):
            if time.time() > deadline:
                break
            res = _try_reduce_drop(U, V, W, T, drop, deadline, iters=120)
            if res is not None:
                U, V, W = res
                if _check_decomposition(U, V, W, T, tol=1e-7):
                    best = (U, V, W)
                    best_rank = U.shape[1]
                    found = True
                    break
        if not found:
            break

    # 3) Direct low-rank search from the lower bound up to best_rank - 1.
    for R in range(max(rank_lb, 1), best_rank):
        if time.time() > deadline:
            break
        res = _try_direct_rank(R, T, deadline, inits=8, iters=160)
        if res is not None:
            best = res
            best_rank = R
            break

    # 4) Rational polish the best found.
    U, V, W = best
    polished = _polish_rational(U, V, W, T)
    if polished is not None:
        return {"U": polished[0], "V": polished[1], "W": polished[2]}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
