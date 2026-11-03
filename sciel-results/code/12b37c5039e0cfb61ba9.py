"""Rewritten program: structured rank search for 3x5x6 matrix multiplication tensor."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 5, 6]
N, M, P = 3, 5, 6
A_ROWS = N * M  # 15
B_ROWS = M * P  # 30
C_ROWS = P * N  # 18


def _build_T():
    T = np.zeros((A_ROWS, B_ROWS, C_ROWS))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T


_T = _build_T()


def _schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(A_ROWS)]
    V = [[0] * rank for _ in range(B_ROWS)]
    W = [[0] * rank for _ in range(C_ROWS)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _matmul(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _residual(U, V, W):
    return np.abs(_matmul(U, V, W) - _T).max()


def _try_complete(U, V, W):
    """Try to complete the last (rank-1) factor to match T exactly.

    U, V, W are lists of (r-1) rank-1 factors. We search for a final
    (a, b, c) such that sum_{l<r} a_l b_l c_l + a b c = T.
    """
    # Build the residual tensor: R = T - sum_{l} a_l b_l c_l
    R = _T.copy()
    for a, b, c in zip(U, V, W):
        R -= np.einsum('i,j,k->ijk', a, b, c)

    # We need a b c = R. For each candidate support on the c-index,
    # solve for a, b.
    # Strategy: pick a set of (i,j) positions in R where R != 0.
    # For a rank-1 tensor, R[i,j,k] = a_i b_j c_k.
    # Fix c_k for k where the column is nonzero, then solve.

    # Find nonzero entries
    nz = np.argwhere(R != 0)
    if len(nz) == 0:
        # R is zero, any a,b,c works (use zeros)
        a = np.zeros(A_ROWS)
        b = np.zeros(B_ROWS)
        c = np.zeros(C_ROWS)
        return a, b, c

    # Try: pick a k0 with a nonzero column, set c = R[:, :, k0] / b_ref
    # Actually for rank-1: R[i,j,k] = a_i b_j c_k
    # So R[:, :, k] = a (b * c_k)^T = c_k * a b^T
    # All columns are scalar multiples of a b^T.

    # Find a representative column
    # For each k, column R[:,:,k] should be c_k * a b^T
    # Compute R[:,:,k] for all k
    # Find k with max norm
    col_norms = np.linalg.norm(R.reshape(A_ROWS, B_ROWS * C_ROWS), axis=1)
    # Actually reshape: R is (A, B, C). Columns over (A,B) for each k.
    col_norms = np.array([np.linalg.norm(R[:, :, k]) for k in range(C_ROWS)])
    k0 = int(np.argmax(col_norms))
    if col_norms[k0] < 1e-12:
        return None, None, None

    # R[:, :, k0] = c_{k0} * a b^T
    # Let a b^T = R[:, :, k0] / c_{k0}. We need to determine c_{k0}.
    # Use another column k1: R[:, :, k1] = c_{k1} * a b^T
    # So R[:, :, k1] = (c_{k1} / c_{k0}) * R[:, :, k0]
    # This gives us the ratio.

    # Pick a, b from the SVD of R[:, :, k0] (up to scale)
    M0 = R[:, :, k0]
    # M0 should be rank 1 (a b^T * c_{k0})
    # Use SVD to get a and b
    try:
        U_svd, s, Vt = np.linalg.svd(M0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None
    if s[0] < 1e-12:
        return None, None, None
    # a = U_svd[:, 0] * sqrt(s[0]) * c_{k0}
    # b = Vt[0, :] * sqrt(s[0]) * c_{k0}
    # But c_{k0} is unknown. Set c_{k0} = 1 for now.
    a = U_svd[:, 0] * s[0]  # this gives a b^T = M0, so a = U * s, b = Vt[0]
    b = Vt[0, :].copy()
    # Now a b^T should equal M0
    # Check: a b^T = M0
    # Now determine c_k for all k: R[:, :, k] = c_k * a b^T
    # c_k = <R[:, :, k], a b^T> / ||a b^T||^2  (if a b^T != 0)
    abT = np.outer(a, b)
    denom = np.sum(abT * abT)
    if denom < 1e-12:
        return None, None, None
    c = np.zeros(C_ROWS)
    for k in range(C_ROWS):
        c[k] = np.sum(R[:, :, k] * abT) / denom

    # Verify
    recon = np.einsum('i,j,k->ijk', a, b, c)
    if np.abs(recon - R).max() < 1e-9:
        return a, b, c
    return None, None, None


def _build_result(U, V, W):
    rank = len(U)
    U_list = [[float(x) for x in u] for u in U]
    V_list = [[float(x) for x in v] for v in V]
    W_list = [[float(x) for x in w] for w in W]
    return {"U": U_list, "V": V_list, "W": W_list}


def _search(rank, seed, time_budget, tol=1e-9):
    """Search for a rank-`rank` decomposition using random rank-1 factors
    and a final completion step."""
    rng = np.random.RandomState(seed)
    best = None
    start = time.time()

    # We'll try to build rank-1 factors one at a time, greedily reducing residual.
    # At each step, pick a random rank-1 factor that best reduces residual,
    # or use random initialization and refine.

    # Strategy: random restarts with random rank-1 factors, then complete.
    n_factors = rank - 1  # we'll complete the last one

    for restart in range(2000):
        if time.time() - start > time_budget - 5:
            break

        # Generate random rank-1 factors
        U = []
        V = []
        W = []
        for _ in range(n_factors):
            a = rng.randn(A_ROWS)
            b = rng.randn(B_ROWS)
            c = rng.randn(C_ROWS)
            U.append(a)
            V.append(b)
            W.append(c)

        a, b, c = _try_complete(U, V, W)
        if a is not None:
            U.append(a)
            V.append(b)
            W.append(c)
            res = _residual(np.array(U), np.array(V), np.array(W))
            if res < tol:
                return _build_result(U, V, W)

    return None


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # Try ranks below 90
    for rank in range(85, 89):  # 85, 86, 87, 88
        remaining = deadline - time.time()
        if remaining < 10:
            break
        budget = min(remaining - 5, 30)
        if budget < 5:
            break
        result = _search(rank, seed=rank, time_budget=budget)
        if result is not None:
            return result

    # Fallback: schoolbook
    return _schoolbook()
