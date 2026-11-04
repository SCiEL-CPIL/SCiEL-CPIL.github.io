"""Matrix multiplication 2,3,9 — rank-minimal exact tensor decomposition.
Vectorized ALS + rank search + high-precision polish + half-integer snap.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n, m, p = DIMENSIONS
    N = n * m          # U rows  = 6
    M = m * p          # V rows  = 27
    P = p * n          # W rows  = 18
    T = n * m * p      # tensor entries = 54

    # Precompute index arrays (row/col) for the three linear systems.
    # For each tensor entry t=(i,j,k):
    #   U-row = i*m+j, V-row = j*p+k, W-row = k*n+i
    i_idx = np.repeat(np.arange(n), m * p)          # length T
    j_idx = np.tile(np.repeat(np.arange(m), p), n)  # length T
    k_idx = np.tile(np.arange(p), n * m)            # length T

    u_row = i_idx * m + j_idx
    v_row = j_idx * p + k_idx
    w_row = k_idx * n + i_idx

    # Build the linear-system structure once per rank r.
    # A_U: (T, N*r), A_U[t, u_row[t]*r + c] = V[v_row[t],c]*W[w_row[t],c]
    # We build the "index" parts (which column each (t,c) maps to) once.
    c_idx = np.tile(np.arange(0), T)  # placeholder, rebuilt per r

    def build_A_struct(r):
        # For A_U: row t, col = u_row[t]*r + c  -> c in [0,r)
        # We'll store per-t the base column offset.
        return u_row, v_row, w_row

    def make_A_U(r, V, W):
        A = np.zeros((T, N * r))
        # column for entry (t,c) is u_row[t]*r + c
        cols = u_row[:, None] * r + np.arange(r)[None, :]   # (T, r)
        # value = V[v_row[t],c] * W[w_row[t],c]
        vals = V[v_row][:, np.arange(r)[None, :]] * W[w_row][:, np.arange(r)[None, :]]  # (T, r)
        A[cols.T.reshape(-1), np.tile(np.arange(N * r), T) * 0 + np.repeat(np.arange(r), T)] = 0  # no-op guard
        # Efficient scatter:
        # flatten row/col
        rflat = cols.reshape(-1)          # (T*r,)
        cflat = np.tile(np.arange(N * r), 0)  # not used
        # build full column index per (t,c): u_row[t]*r + c
        col_idx = (u_row[:, None] * r + np.arange(r)[None, :]).reshape(-1)
        row_idx = np.repeat(np.arange(T), r)
        val_flat = (V[v_row][:, np.arange(r)[None, :]] * W[w_row][:, np.arange(r)[None, :]]).reshape(-1)
        A[row_idx, col_idx] = val_flat
        return A

    def make_A_V(r, U, W):
        A = np.zeros((T, M * r))
        col_idx = (v_row[:, None] * r + np.arange(r)[None, :]).reshape(-1)
        row_idx = np.repeat(np.arange(T), r)
        val_flat = (U[u_row][:, np.arange(r)[None, :]] * W[w_row][:, np.arange(r)[None, :]]).reshape(-1)
        A[row_idx, col_idx] = val_flat
        return A

    def make_A_W(r, U, V):
        A = np.zeros((T, P * r))
        col_idx = (w_row[:, None] * r + np.arange(r)[None, :]).reshape(-1)
        row_idx = np.repeat(np.arange(T), r)
        val_flat = (U[u_row][:, np.arange(r)[None, :]] * V[v_row][:, np.arange(r)[None, :]]).reshape(-1)
        A[row_idx, col_idx] = val_flat
        return A

    b = np.ones(T)

    def residual(U, V, W):
        r = U.shape[1]
        # sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        prod = U[u_row] * V[v_row] * W[w_row]  # (T, r)
        s = prod.sum(axis=1)
        return np.max(np.abs(s - 1.0))

    def verify(U, V, W, tol=1e-9):
        return residual(U, V, W) <= tol

    # Schoolbook fallback (rank 54)
    U_sb = np.zeros((N, T)); V_sb = np.zeros((M, T)); W_sb = np.zeros((P, T))
    for t in range(T):
        U_sb[u_row[t], t] = 1.0
        V_sb[v_row[t], t] = 1.0
        W_sb[w_row[t], t] = 1.0

    best_rank = T
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def als_solve(r, max_iters, tol, seed):
        rng = np.random.RandomState(seed)
        U = rng.randn(N, r) * 0.5
        V = rng.randn(M, r) * 0.5
        W = rng.randn(P, r) * 0.5
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            A = make_A_U(r, V, W)
            Uf, *_ = np.linalg.lstsq(A, b, rcond=None)
            U = Uf.reshape(N, r)
            A = make_A_V(r, U, W)
            Vf, *_ = np.linalg.lstsq(A, b, rcond=None)
            V = Vf.reshape(M, r)
            A = make_A_W(r, U, V)
            Wf, *_ = np.linalg.lstsq(A, b, rcond=None)
            W = Wf.reshape(P, r)
            err = residual(U, V, W)
            if err < tol:
                return U, V, W, err
        return U, V, W, residual(U, V, W)

    def polish(U, V, W, max_iters=300, tol=1e-14):
        r = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            A = make_A_U(r, V, W); U = np.linalg.lstsq(A, b, rcond=None)[0].reshape(N, r)
            A = make_A_V(r, U, W); V = np.linalg.lstsq(A, b, rcond=None)[0].reshape(M, r)
            A = make_A_W(r, U, V); W = np.linalg.lstsq(A, b, rcond=None)[0].reshape(P, r)
            if residual(U, V, W) < tol:
                break
        return U, V, W, residual(U, V, W)

    # Search ranks from low to high.
    for r in range(6, T + 1):
        if time.time() > deadline - 20:
            break
        found = False
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            U, V, W, err = als_solve(r, max_iters=120, tol=1e-11, seed=seed)
            if err < 1e-9:
                U, V, W, err = polish(U, V, W)
                if err < 1e-8:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        if found:
            break

    # Half-integer snap with strict verification.
    def snap(arr):
        return np.round(arr * 2.0) / 2.0

    U_h, V_h, W_h = snap(best_U), snap(best_V), snap(best_W)
    if verify(U_h, V_h, W_h, tol=1e-6):
        best_U, best_V, best_W = U_h, V_h, W_h

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
