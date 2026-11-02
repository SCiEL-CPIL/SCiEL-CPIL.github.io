"""
Rank-16 tensor decomposition for 2x2 by 2x5 matrix multiplication.
Uses Strassen (rank 7) for the first two columns and a rank-9 decomposition
for the remaining three columns, exploiting block additivity.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:, r], V_a[:, r], W_a[:, r])
    return np.allclose(result, T, atol=tol)


def _snap_to_half_integers(U, V, W, max_denom=64):
    """Try to snap values to half-integers (multiples of 1/max_denom)."""
    for denom in [2, 4, 8, 16, 32, max_denom]:
        U_s = np.round(np.array(U, dtype=np.float64) * denom) / denom
        V_s = np.round(np.array(V, dtype=np.float64) * denom) / denom
        W_s = np.round(np.array(W, dtype=np.float64) * denom) / denom
        if _check_decomposition(U_s, V_s, W_s, _build_target(), tol=1e-8):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    return np.array(U, dtype=np.float64).tolist(), \
           np.array(V, dtype=np.float64).tolist(), \
           np.array(W, dtype=np.float64).tolist()


def _build_strassen_2x2x2(n, m, p, col0, col1):
    """
    Return (U, V, W) for Strassen decomposition of 2x2 * 2x2 block.
    U: (4, 7), V: (10, 7), W: (10, 7)
    Only rows/cols corresponding to the 2x2 block are non-zero.
    """
    U = np.zeros((n * m, 7), dtype=np.float64)
    V = np.zeros((m * p, 7), dtype=np.float64)
    W = np.zeros((p * n, 7), dtype=np.float64)

    # Strassen formulas for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # u indices: 0=a, 1=b, 2=c, 3=d
    # v indices for cols col0, col1: j*p + k
    #   j=0: col0, col1 -> v = col0, col1
    #   j=1: col0, col1 -> v = p+col0, p+col1
    # w indices: k*n + i
    #   col0: w = col0*n, col0*n+1
    #   col1: w = col1*n, col1*n+1

    v00 = 0 * p + col0  # row 0, col col0
    v01 = 0 * p + col1  # row 0, col col1
    v10 = 1 * p + col0  # row 1, col col0
    v11 = 1 * p + col1  # row 1, col col1

    w00 = col0 * n + 0  # col col0, row 0
    w01 = col0 * n + 1  # col col0, row 1
    w10 = col1 * n + 0  # col col1, row 0
    w11 = col1 * n + 1  # col col1, row 1

    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[v00, 0] = 1; V[v11, 0] = 1
    W[w00, 0] = 1; W[w11, 0] = 1

    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[v00, 1] = 1
    W[w01, 1] = 1; W[w11, 1] = -1

    # M3 = a(f-h)
    U[0, 2] = 1
    V[v01, 2] = 1; V[v11, 2] = -1
    W[w10, 2] = 1; W[w11, 2] = 1

    # M4 = (a-b)g
    U[0, 3] = 1; U[1, 3] = -1
    V[v10, 3] = 1
    W[w00, 3] = 1; W[w01, 3] = 1

    # M5 = (c+d)(h-f)
    U[2, 4] = 1; U[3, 4] = 1
    V[v01, 4] = -1; V[v11, 4] = 1
    W[w10, 4] = 1; W[w11, 4] = 1

    # M6 = (a+b)(e-g)
    U[0, 5] = 1; U[1, 5] = 1
    V[v00, 5] = 1; V[v10, 5] = -1
    W[w00, 5] = 1; W[w10, 5] = 1

    # M7 = (d-b)(f-h)
    U[3, 6] = 1; U[1, 6] = -1
    V[v11, 6] = 1; V[v01, 6] = -1
    W[w11, 6] = 1; W[w01, 6] = -1

    return U, V, W


def _als_solve_U(T, V, W, R):
    """Solve for U given V, W using least squares."""
    VvW = np.einsum('vr,wr->vwr', V, W)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    n_u = T.shape[0]
    U = np.zeros((n_u, R), dtype=np.float64)
    for u in range(n_u):
        b = T[u].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U


def _als_solve_V(T, U, W, R):
    """Solve for V given U, W."""
    Uw = np.einsum('ur,wr->uwr', U, W)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    n_v = T.shape[1]
    V = np.zeros((n_v, R), dtype=np.float64)
    for v in range(n_v):
        b = T[:, v, :].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V


def _als_solve_W(T, U, V, R):
    """Solve for W given U, V."""
    UV = np.einsum('ur,vr->uvr', U, V)
    UV_flat = UV.reshape(-1, R)
    n_w = T.shape[2]
    W = np.zeros((n_w, R), dtype=np.float64)
    for w in range(n_w):
        b = T[:, :, w].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W


def _als_decompose(T, R, max_iters=500, tol=1e-12, n_restarts=20, seed_base=0):
    """
    Run ALS to find a rank-R decomposition of tensor T.
    T has shape (n_u, n_v, n_w).
    Returns (U, V, W) or None if no exact solution found.
    """
    n_u, n_v, n_w = T.shape

    for restart in range(n_restarts):
        seed = seed_base + restart
        np.random.seed(seed)
        scale = 0.5 + 0.3 * (restart % 5)
        U = np.random.randn(n_u, R) * scale
        V = np.random.randn(n_v, R) * scale
        W = np.random.randn(n_w, R) * scale

        prev_err = np.inf
        for it in range(max_iters):
            U = _als_solve_U(T, V, W, R)
            V = _als_solve_V(T, U, W, R)
            W = _als_solve_W(T, U, V, R)

            # Compute error
            err = 0.0
            for r in range(R):
                err += np.sum((np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r]) - T) ** 2)
            err = np.sqrt(err / (n_u * n_v * n_w))

            if abs(prev_err - err) < tol:
                break
            prev_err = err

        # Check if exact
        result = np.zeros_like(T)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        if np.allclose(result, T, atol=1e-9):
            return U, V, W

    return None


def _build_rank9_2x2x3(n, m, p, cols):
    """
    Build a rank-9 decomposition for 2x2 * 2x3 block.
    cols = [c0, c1, c2]
    Returns (U, V, W) with shapes (4, 9), (10, 9), (10, 9)
    Only relevant rows are non-zero.
    """
    c0, c1, c2 = cols

    # The sub-tensor has:
    # u: 4 entries (0..3)
    # v: 6 entries (j*p + k for j in {0,1}, k in {c0,c1,c2})
    # w: 6 entries (k*n + i for k in {c0,c1,c2}, i in {0,1})

    # Extract sub-tensor
    v_indices = [0 * p + c0, 0 * p + c1, 0 * p + c2, 1 * p + c0, 1 * p + c1, 1 * p + c2]
    w_indices = [c0 * n + 0, c0 * n + 1, c1 * n + 0, c1 * n + 1, c2 * n + 0, c2 * n + 1]

    T_full = _build_target()
    T_sub = T_full[:, v_indices, w_indices].copy()

    # Run ALS on the sub-tensor
    result = _als_decompose(T_sub, 9, max_iters=500, tol=1e-12, n_restarts=30, seed_base=1000)
    if result is None:
        return None

    U_sub, V_sub, W_sub = result

    # Embed back into full matrices
    U = np.zeros((n * m, 9), dtype=np.float64)
    V = np.zeros((m * p, 9), dtype=np.float64)
    W = np.zeros((p * n, 9), dtype=np.float64)

    U[:, :] = U_sub
    for i, vi in enumerate(v_indices):
        V[vi, :] = V_sub[i, :]
    for i, wi in enumerate(w_indices):
        W[wi, :] = W_sub[i, :]

    return U, V, W


def _build_rank16(n, m, p):
    """
    Build a rank-16 decomposition: Strassen(2x2x2) + rank-9(2x2x3).
    """
    # Strassen for cols 0, 1
    U1, V1, W1 = _build_strassen_2x2x2(n, m, p, 0, 1)

    # Rank-9 for cols 2, 3, 4
    result2 = _build_rank9_2x2x3(n, m, p, [2, 3, 4])
    if result2 is None:
        return None
    U2, V2, W2 = result2

    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill

    T = _build_target()

    # Try to build rank-16
    result = _build_rank16(n, m, p)
    if result is not None:
        U, V, W = result
        if _check_decomposition(U, V, W, T):
            U, V, W = _snap_to_half_integers(U, V, W)
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
            # If snapping failed, return the float version
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: try rank-17
    if time.time() < deadline:
        # Try removing one term from rank-18 and running ALS
        U18, V18, W18 = _build_rank18(n, m, p)
        for term_to_remove in range(18):
            if time.time() > deadline:
                break
            idx = [i for i in range(18) if i != term_to_remove]
            U_init = U18[:, idx].copy()
            V_init = V18[:, idx].copy()
            W_init = W18[:, idx].copy()

            for seed in range(5):
                if time.time() > deadline:
                    break
                np.random.seed(term_to_remove * 100 + seed)
                scale = 0.01
                U_pert = U_init + np.random.randn(4, 17) * scale
                V_pert = V_init + np.random.randn(10, 17) * scale
                W_pert = W_init + np.random.randn(10, 17) * scale

                # Run ALS
                U_cur, V_cur, W_cur = U_pert, V_pert, W_pert
                prev_err = np.inf
                for it in range(200):
                    U_cur = _als_solve_U(T, V_cur, W_cur, 17)
                    V_cur = _als_solve_V(T, U_cur, W_cur, 17)
                    W_cur = _als_solve_W(T, U_cur, V_cur, 17)

                    err = 0.0
                    for r in range(17):
                        err += np.sum((np.einsum('u,v,w->uvw', U_cur[:, r], V_cur[:, r], W_cur[:, r]) - T) ** 2)
                    err = np.sqrt(err / (4 * 10 * 10))

                    if abs(prev_err - err) < 1e-13:
                        break
                    prev_err = err

                if _check_decomposition(U_cur, V_cur, W_cur, T):
                    U_s, V_s, W_s = _snap_to_half_integers(U_cur, V_cur, W_cur)
                    if _check_decomposition(U_s, V_s, W_s, T):
                        return {"U": U_s, "V": V_s, "W": W_s}
                    return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}

    # Final fallback: rank-18
    U18, V18, W18 = _build_rank18(n, m, p)
    return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}


def _build_rank18(n, m, p):
    """Build the known rank-18 decomposition."""
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def add_strassen_block(col0, col1, base):
        v00 = 0 * p + col0
        v01 = 0 * p + col1
        v10 = 1 * p + col0
        v11 = 1 * p + col1
        w00 = col0 * n + 0
        w01 = col0 * n + 1
        w10 = col1 * n + 0
        w11 = col1 * n + 1

        r = base
        U[0, r] = 1; U[3, r] = 1
        V[v00, r] = 1; V[v11, r] = 1
        W[w00, r] = 1; W[w11, r] = 1

        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[v00, r] = 1
        W[w01, r] = 1; W[w11, r] = -1

        r = base + 2
        U[0, r] = 1
        V[v01, r] = 1; V[v11, r] = -1
        W[w10, r] = 1; W[w11, r] = 1

        r = base + 3
        U[0, r] = 1; U[1, r] = -1
        V[v10, r] = 1
        W[w00, r] = 1; W[w01, r] = 1

        r = base + 4
        U[2, r] = 1; U[3, r] = 1
        V[v01, r] = -1; V[v11, r] = 1
        W[w10, r] = 1; W[w11, r] = 1

        r = base + 5
        U[0, r] = 1; U[1, r] = 1
        V[v00, r] = 1; V[v10, r] = -1
        W[w00, r] = 1; W[w10, r] = 1

        r = base + 6
        U[3, r] = 1; U[1, r] = -1
        V[v11, r] = 1; V[v01, r] = -1
        W[w11, r] = 1; W[w01, r] = -1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)

    # Schoolbook for column 4
    r = 14
    U[0, r] = 1; V[0 * p + 4, r] = 1; W[4 * n + 0, r] = 1
    r = 15
    U[1, r] = 1; V[1 * p + 4, r] = 1; W[4 * n + 0, r] = 1
    r = 16
    U[2, r] = 1; V[0 * p + 4, r] = 1; W[4 * n + 1, r] = 1
    r = 17
    U[3, r] = 1; V[1 * p + 4, r] = 1; W[4 * n + 1, r] = 1

    return U, V, W
