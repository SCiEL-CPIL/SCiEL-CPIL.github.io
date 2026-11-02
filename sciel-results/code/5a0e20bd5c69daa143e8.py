"""Tensor decomposition search for 2x3x10 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _build_tensor(n, m, p):
    """Build the n x m x p matrix multiplication tensor (flattened)."""
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    return tensor

def _verify_decomposition(U, V, W, tensor, tol=1e-9):
    """Verify that U, V, W decompose the tensor."""
    T = np.zeros_like(tensor)
    rank = U.shape[1]
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T, tensor, atol=tol)

def _verify_half_integer(U, V, W, tensor, tol=1e-9):
    """Verify decomposition AND that entries are integer or half-integer."""
    for M in [U, V, W]:
        # Check each entry is within tol of a half-integer
        rounded = np.round(M * 2) / 2
        if not np.allclose(M, rounded, atol=tol):
            return False
    return _verify_decomposition(U, V, W, tensor, tol)

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _als_step(T, U, V, W, r, deadline):
    """One alternating least squares iteration. Returns (U, V, W, error)."""
    nm, mp, pn = T.shape
    # Solve for U: T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # Flatten (j,k) axis: T_reshaped[i, jk] = sum_r U[i,r] * (VW)[jk, r]
    VW = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, r)
    T_u = T.reshape(nm, mp * pn)
    # U = T_u @ pinv(VW)
    try:
        U_new = T_u @ np.linalg.pinv(VW)
    except np.linalg.LinAlgError:
        U_new = U
    if time.time() > deadline:
        return U_new, V, W, None

    # Solve for V
    UW = np.einsum('ir,kr->ikr', U_new, W).reshape(nm * pn, r)
    T_v = T.reshape(mp, nm * pn).T  # T_v[jk, i] -> need T[j, i, k]
    # Actually T[j, i, k] with j from mp, so reshape T to (mp, nm*pn)
    T_v = T.reshape(mp, nm * pn)
    try:
        V_new = T_v @ np.linalg.pinv(UW)
    except np.linalg.LinAlgError:
        V_new = V
    if time.time() > deadline:
        return U_new, V_new, W, None

    # Solve for W
    UV = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, r)
    T_w = T.reshape(pn, nm * mp)
    try:
        W_new = T_w @ np.linalg.pinv(UV)
    except np.linalg.LinAlgError:
        W_new = W

    T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
    error = np.max(np.abs(T_check - T))
    return U_new, V_new, W_new, error

def _try_rank(T, r, n, m, p, U_school, V_school, W_school, deadline, max_seeds=8, max_iters=15):
    """Try to find a valid rank-r decomposition."""
    nm, mp, pn = T.shape
    best = None
    best_err = np.inf

    for seed in range(max_seeds):
        if time.time() > deadline:
            break
        np.random.seed(seed * 1337 + r)

        # Different initialization strategies
        if seed % 4 == 0:
            # Truncated schoolbook
            if r <= n*m*p:
                U_c = U_school[:, :r].copy()
                V_c = V_school[:, :r].copy()
                W_c = W_school[:, :r].copy()
            else:
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
        elif seed % 4 == 1:
            # Random
            U_c = np.random.randn(nm, r)
            V_c = np.random.randn(mp, r)
            W_c = np.random.randn(pn, r)
        elif seed % 4 == 2:
            # Sparse random
            U_c = np.random.randn(nm, r)
            V_c = np.random.randn(mp, r)
            W_c = np.random.randn(pn, r)
            U_c[np.random.rand(nm, r) < 0.4] = 0
            V_c[np.random.rand(mp, r) < 0.4] = 0
            W_c[np.random.rand(pn, r) < 0.4] = 0
        else:
            # Structured: exploit the 2x3 geometry
            U_c = np.zeros((nm, r))
            V_c = np.zeros((mp, r))
            W_c = np.zeros((pn, r))
            for col in range(r):
                i1 = col % n
                j1 = (col // n) % m
                k2 = (col // (n * m)) % p
                U_c[i1*m + j1, col] = 1.0
                V_c[j1*p + k2, col] = 1.0
                W_c[k2*n + i1, col] = 1.0
            # Add small perturbation
            U_c += 0.1 * np.random.randn(nm, r)
            V_c += 0.1 * np.random.randn(mp, r)
            W_c += 0.1 * np.random.randn(pn, r)

        # ALS iterations
        for it in range(max_iters):
            if time.time() > deadline:
                break
            U_c, V_c, W_c, err = _als_step(T, U_c, V_c, W_c, r, deadline)
            if err is None:
                break
            if err < 1e-8:
                break

        # Check if we got a good solution
        T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
        err = np.max(np.abs(T_check - T))
        if err < best_err:
            best_err = err
            best = (U_c.copy(), V_c.copy(), W_c.copy())

        # If we found a very good solution, try to round to half-integers
        if err < 1e-6:
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2
            T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            err2 = np.max(np.abs(T_check2 - T))
            if err2 < 1e-10:
                return U_r, V_r, W_r

    # If best solution is good but not half-integer, try rounding
    if best is not None and best_err < 1e-5:
        U_c, V_c, W_c = best
        U_r = np.round(U_c * 2) / 2
        V_r = np.round(V_c * 2) / 2
        W_r = np.round(W_c * 2) / 2
        T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        err2 = np.max(np.abs(T_check2 - T))
        if err2 < 1e-10:
            return U_r, V_r, W_r

    return None

def construct():
    n, m, p = DIMENSIONS
    tensor = _build_tensor(n, m, p)

    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]

    deadline = time.time() + 130  # 20s safety margin

    # Try to find lower rank decompositions
    # The tensor has a special structure: for each (i, j, k), there's exactly one 1
    # at position (i*m+j, j*p+k, k*n+i)
    # This is a permutation-like tensor

    # Try ranks from 59 down
    for r in range(best_rank - 1, max(n*m, m*p, p*n) - 1, -1):
        if time.time() > deadline:
            break

        result = _try_rank(tensor, r, n, m, p, U, V, W, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            # Verify
            if _verify_half_integer(U_r, V_r, W_r, tensor):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
