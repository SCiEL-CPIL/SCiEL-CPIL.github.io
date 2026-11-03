"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]
DEADLINE = 140.0  # safety margin under the 150s kill

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _to_np(res):
    return (np.array(res["U"], dtype=float),
            np.array(res["V"], dtype=float),
            np.array(res["W"], dtype=float))

def _verify_np(U, V, W):
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    prod = U * V.T  # (N,M)
    T = np.einsum('ij,jk->ik', prod, W.T)  # (N,P)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (j == k) else 0.0
                if abs(T[i * m + j, k * n + i] - expected) > 1e-6:
                    return False
    return True

def _verify(res):
    U, V, W = _to_np(res)
    return _verify_np(U, V, W)

def _search():
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    rng = np.random.default_rng(12345)

    # Build target tensor T (N x P) with T[i*m+j, k*n+i] = 1 if j==k else 0
    T = np.zeros((N, P), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T[i * m + j, k * n + i] = 1.0

    def refine(U, V, W, iters=60, tol=1e-9):
        """Alternating least squares: minimize ||U V^T - T W^T||_F^2."""
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        # target for W: T = (U V^T) W^T  =>  W^T = (U V^T)^+ T^T
        # target for U: T W^T = U (V^T W^T) => U = (T W^T)(V^T W^T)^+
        # target for V: (U^T T) = (U^T W) V^T => V^T = (U^T W)^+ (U^T T)
        for _ in range(iters):
            # Update W
            A = U @ V.T                      # (N,M)
            # T = A @ W.T  => W.T = A^+ T.T
            Wt, *_ = np.linalg.lstsq(A, T.T, rcond=None)
            W = Wt.T
            # Update U
            B = T @ W.T                      # (N,P)
            C = V @ W.T                      # (M,P)
            U, *_ = np.linalg.lstsq(C.T, B.T, rcond=None)
            U = U.T
            # Update V
            D = U.T @ T                      # (M,N)
            E = U.T @ W                      # (M,P)
            Vt, *_ = np.linalg.lstsq(E, D.T, rcond=None)
            V = Vt.T
        return U, V, W

    def round_halfint(U, V, W):
        """Round to nearest half-integer and re-verify."""
        Ur = np.round(U * 2) / 2
        Vr = np.round(V * 2) / 2
        Wr = np.round(W * 2) / 2
        if _verify_np(Ur, Vr, Wr):
            return Ur, Vr, Wr
        # try rounding each independently with small search
        return None

    best_rank = None
    best_res = None

    # Try a range of ranks
    for rank in range(60, 97):
        if time.time() > DEADLINE:
            break
        # Several random restarts
        max_restarts = 4
        for r in range(max_restarts):
            if time.time() > DEADLINE:
                break
            # Random initialization with some structure
            U = rng.normal(size=(N, rank))
            V = rng.normal(size=(M, rank))
            W = rng.normal(size=(P, rank))
            # Scale
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)

            U, V, W = refine(U, V, W, iters=80)

            # Check residual
            A = U @ V.T
            resid = np.linalg.norm(A @ W.T - T)
            if resid < 1e-5:
                # Try to find exact half-integer solution
                res = round_halfint(U, V, W)
                if res is not None:
                    Ur, Vr, Wr = res
                    best_rank = rank
                    best_res = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    break
            if best_rank is not None:
                break
        if best_rank is not None:
            break

    return best_res

def construct():
    start = time.time()
    # Fallback: schoolbook rank 96
    result = _schoolbook()
    assert _verify(result), "schoolbook should verify"

    # Try to find better
    if time.time() < DEADLINE - 5:
        better = _search()
        if better is not None and _verify(better):
            result = better

    # Ensure we return before deadline
    if time.time() > DEADLINE - 2:
        return result
    return result
