# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims.
    Returns X with rows centered, shape (n, k)."""
    n = M.shape[0]
    # Double centering: B = -1/2 J M J
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    # Eigendecomposition (B symmetric)
    w, V = np.linalg.eigh(B)
    # Take largest k eigenvalues
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    # X = V * sqrt(w)
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    """Find permutation p minimizing sum ||X[i] - Y[p[i]]||^2 via Hungarian-like greedy.
    Actually solve the assignment: minimize sum_{i} ||X[i] - Y[p[i]]||^2.
    This is a linear assignment problem on the cost matrix C[i,j] = ||X[i]-Y[j]]||^2.
    Use scipy's linear_sum_assignment."""
    from scipy.optimize import linear_sum_assignment
    n = X.shape[0]
    # C[i,j] = ||X[i] - Y[j]]||^2 = ||X[i]]||^2 - 2 X[i].Y[j] + ||Y[j]]||^2
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    # linear_sum_assignment minimizes sum C[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _two_opt_search(a, b, p, time_limit, max_iter=500000):
    """Vectorized 2-opt local search. p is numpy int64 array (facility i -> location p[i]).
    We try swapping locations of two facilities i,j and accept if cost decreases."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    # Precompute RC and CC for fast delta computation
    # RC[l, k] = b[l, p[k]]
    # CC[k, l] = b[p[k], l]
    RC = b[:, p]  # shape (n, n)
    CC = b[p, :]  # shape (n, n)

    # a_row_diff[i, j] = a[i, :] - a[j, :]  -- too big (n^2 * n), compute on the fly
    # Instead, precompute a as float for speed
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)

    start = time.time()
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                li = p[i]
                lj = p[j]
                # Compute delta using precomputed RC, CC
                # drow = RC[lj, :] - RC[li, :]
                # dcol = CC[:, lj] - CC[:, li]
                # delta = (a[i,:] - a[j,:]) @ drow + (a[:,i] - a[:,j]) @ dcol
                drow = RC[lj, :] - RC[li, :]
                dcol = CC[:, lj] - CC[:, li]
                a_diff_row = a_float[i, :] - a_float[j, :]
                a_diff_col = a_float[:, i] - a_float[:, j]
                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
                if delta < 0:
                    # Accept swap
                    p[i], p[j] = lj, li
                    cost += delta
                    # Update RC and CC
                    # RC[:, i] was b[li, p[i]] = b[li, li]... wait no.
                    # RC[l, k] = b[l, p[k]]. After swap, p[i]=lj, p[j]=li.
                    # RC[:, i] = b[:, lj] (was b[:, li])
                    # RC[:, j] = b[:, li] (was b[:, lj])
                    RC[:, i] = b_float[:, lj]
                    RC[:, j] = b_float[:, li]
                    # CC[k, l] = b[p[k], l]. After swap, p[i]=lj, p[j]=li.
                    # CC[i, :] = b[lj, :] (was b[li, :])
                    # CC[j, :] = b[li, :] (was b[lj, :])
                    CC[i, :] = b_float[lj, :]
                    CC[j, :] = b_float[li, :]
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    improved = True
        if not improved:
            break
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    # Step 1: MDS embedding + Procrustes alignment
    # Symmetrize for MDS
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0

    # Choose embedding dimension
    k = min(20, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)

    # Procrustes alignment
    p = _procrustes_align(X, Y)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    # Step 2: Local search polish
    time_left = deadline - time.time()
    if time_left > 5:
        p, cost = _two_opt_search(a, b, p, time_left)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p.tolist()
# EVOLVE-BLOCK-END
