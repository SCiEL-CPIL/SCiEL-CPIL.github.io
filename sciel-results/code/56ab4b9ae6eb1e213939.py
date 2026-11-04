import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 8]


def _check(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is valid."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err < tol, float(err)


def _trim(U, V, W):
    """Remove all-zero columns."""
    n_u, n_v, n_w = len(U), len(V), len(W)
    rank = len(U[0])
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2


def _to_arrays(U, V, W):
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)


def _schoolbook_rank32():
    """8 independent schoolbook 2x2 blocks, rank 32."""
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for l in range(4):
        for rb in range(2):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = col
                        U[i * m + (rb * 2 + j), r] = 1
                        V[(rb * 2 + j) * p + (2 * l + k), r] = 1
                        W[(2 * l + k) * n + i, r] = 1
                        col += 1
    return U, V, W


def _strassen_rank56():
    """Original Strassen-based rank-56 decomposition."""
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    strassen = [
        {(0, 0): 1, (1, 1): 1, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {(1, 0): 1, (1, 1): 1, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {(0, 0): 1, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {(1, 1): 1, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {(0, 0): 1, (0, 1): 1, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {(1, 0): 1, (0, 0): -1, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {(0, 1): 1, (1, 1): -1, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term.items():
                    if not isinstance(c, dict):
                        j = j_high * 2 + j_low
                        U[i * m + j, col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k, col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i, col] += c
                col += 1
    return U, V, W


def _find_dependent_columns(U, V, W, tol=1e-10):
    """
    Find columns that are linear combinations of others.
    Returns list of column indices that can be removed.
    """
    r = U.shape[1]
    if r <= 1:
        return []

    # Build the tensor as a matrix: each column r maps to a vector in R^(n*m + m*p + p*n)
    # Actually, for rank reduction we need: column r is redundant if the 3-tensor
    # contribution can be expressed as a linear combination of other columns.
    #
    # A column r is redundant if there exist coefficients c_s (s != r) such that:
    #   U[:,r] = sum_s c_s * U[:,s]  AND  V[:,r] = sum_s c_s * V[:,s]  AND  W[:,r] = sum_s c_s * W[:,s]
    #
    # This is a system of (n*m + m*p + p*n) equations in (r-1) unknowns.
    # Stack U, V, W into one big matrix M of shape (n*m+m*p+p*n, r).
    # Column r is in the span of others iff the rank of M doesn't drop when we remove column r.

    n_u, n_v, n_w = U.shape[0], V.shape[0], W.shape[0]
    M = np.vstack([U, V, W])  # shape (n_u+n_v+n_w, r)

    # Compute rank of M
    rank_M = np.linalg.matrix_rank(M, tol=tol)

    # Try removing each column
    removable = []
    for c in range(r):
        M_no_c = np.delete(M, c, axis=1)
        rank_no_c = np.linalg.matrix_rank(M_no_c, tol=tol)
        if rank_no_c == rank_M:
            removable.append(c)

    return removable


def _null_space_reduce(U, V, W, tol=1e-10):
    """
    Aggressive rank reduction: iteratively find and remove dependent columns.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    removed = set()
    changed = True
    while changed:
        changed = False
        removable = _find_dependent_columns(U, V, W, tol)
        for c in removable:
            if c not in removed:
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                removed.add(c)
                changed = True
    return U, V, W


def _local_search_reduce(U, V, W, n, m, p, max_iters=50, tol=1e-9):
    """
    Try to reduce rank by zeroing out entries and checking validity.
    Greedy: for each column, try to zero out one entry at a time.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]

    for _ in range(max_iters):
        improved = False
        # Try removing each column
        for c in range(r - 1, -1, -1):
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            ok, _ = _check(U_test, V_test, W_test, n, m, p, tol)
            if ok:
                U, V, W = U_test, V_test, W_test
                r -= 1
                improved = True
                break
        if not improved:
            break
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130  # safety margin

    best_rank = None
    best_result = None

    # Start with schoolbook rank-32
    U, V, W = _schoolbook_rank32()
    ok, err = _check(U, V, W, n, m, p)
    if ok:
        best_rank = U.shape[1]
        best_result = (U.copy(), V.copy(), W.copy())

    # Try null-space reduction on rank-32
    if time.time() < deadline - 30:
        U2, V2, W2 = _null_space_reduce(U, V, W)
        ok2, err2 = _check(U2, V2, W2, n, m, p)
        if ok2 and U2.shape[1] < best_rank:
            best_rank = U2.shape[1]
            best_result = (U2.copy(), V2.copy(), W2.copy())

    # Try local search on the reduced result
    if time.time() < deadline - 20:
        U3, V3, W3 = _local_search_reduce(best_result[0], best_result[1], best_result[2], n, m, p, max_iters=30)
        ok3, err3 = _check(U3, V3, W3, n, m, p)
        if ok3 and U3.shape[1] < best_rank:
            best_rank = U3.shape[1]
            best_result = (U3.copy(), V3.copy(), W3.copy())

    # Also try from Strassen rank-56 with null-space reduction
    if time.time() < deadline - 30:
        U56, V56, W56 = _strassen_rank56()
        ok56, _ = _check(U56, V56, W56, n, m, p)
        if ok56:
            U56r, V56r, W56r = _null_space_reduce(U56, V56, W56)
            ok56r, _ = _check(U56r, V56r, W56r, n, m, p)
            if ok56r and U56r.shape[1] < best_rank:
                best_rank = U56r.shape[1]
                best_result = (U56r.copy(), V56r.copy(), W56r.copy())

    # Convert back to list format
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: return schoolbook
    U, V, W = _schoolbook_rank32()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
