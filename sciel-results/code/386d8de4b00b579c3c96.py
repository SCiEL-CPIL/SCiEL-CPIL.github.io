"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin under 150s wall-clock

    n, m, p = 2, 5, 5
    N = n * m * p  # 50 total constraints
    U_rows = n * m  # 10
    V_rows = m * p  # 25
    W_rows = p * n  # 10

    def build_schoolbook(rank):
        U = np.zeros((U_rows, rank))
        V = np.zeros((V_rows, rank))
        W = np.zeros((W_rows, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    def compute_residual(U, V, W):
        R = U.shape[1]
        res = np.zeros(N)
        idx = 0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    res[idx] = np.dot(u_row, v_row * w_row) - 1.0
                    idx += 1
        return res

    def max_abs_residual(U, V, W):
        return float(np.max(np.abs(compute_residual(U, V, W))))

    def solve_for_U(V, W, R):
        U = np.zeros((U_rows, R))
        for i in range(n):
            for j in range(m):
                A_block = np.zeros((p, R))
                b_block = np.ones(p)
                for k in range(p):
                    A_block[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                    U[i*m+j, :] = u_row
                except np.linalg.LinAlgError:
                    U[i*m+j, :] = 0
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((V_rows, R))
        for j in range(m):
            for k in range(p):
                A_block = np.zeros((n, R))
                b_block = np.ones(n)
                for i in range(n):
                    A_block[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                    V[j*p+k, :] = v_row
                except np.linalg.LinAlgError:
                    V[j*p+k, :] = 0
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((W_rows, R))
        for k in range(p):
            for i in range(n):
                A_block = np.zeros((m, R))
                b_block = np.ones(m)
                for j in range(m):
                    A_block[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                    W[k*n+i, :] = w_row
                except np.linalg.LinAlgError:
                    W[k*n+i, :] = 0
        return W

    def build_jacobian(U, V, W):
        """Build Jacobian efficiently using precomputed index arrays."""
        R = U.shape[1]
        total_vars = U.size + V.size + W.size
        J = np.zeros((N, total_vars))
        idx = 0
        for i in range(n):
            for j in range(m):
                u_idx = i*m+j
                for k in range(p):
                    v_idx = j*p+k
                    w_idx = k*n+i
                    # Vectorized per-constraint row
                    J[idx, u_idx*R:(u_idx+1)*R] = V[v_idx, :] * W[w_idx, :]
                    J[idx, U.size + v_idx*R:U.size + (v_idx+1)*R] = U[u_idx, :] * W[w_idx, :]
                    J[idx, U.size + V.size + w_idx*R:U.size + V.size + (w_idx+1)*R] = U[u_idx, :] * V[v_idx, :]
                    idx += 1
        return J

    def gauss_newton(U, V, W, max_iter=500, tol=1e-11):
        R = U.shape[1]
        total_vars = U.size + V.size + W.size
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            res = compute_residual(U, V, W)
            max_res = np.max(np.abs(res))
            if max_res < tol:
                break
            J = build_jacobian(U, V, W)
            try:
                JtJ = J.T @ J
                JtJ += 1e-8 * np.eye(total_vars)
                delta = np.linalg.solve(JtJ, -J.T @ res)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(JtJ, -J.T @ res, rcond=None)[0]
                except Exception:
                    break
            # Line search for stability
            alpha = 1.0
            for _ in range(20):
                Uf = U.flatten() + alpha * delta[:U.size]
                Vf = V.flatten() + alpha * delta[U.size:U.size+V.size]
                Wf = W.flatten() + alpha * delta[U.size+V.size:]
                U_trial = Uf.reshape(U.shape)
                V_trial = Vf.reshape(V.shape)
                W_trial = Wf.reshape(W.shape)
                new_res = compute_residual(U_trial, V_trial, W_trial)
                if np.max(np.abs(new_res)) < max_res:
                    break
                alpha *= 0.5
            U = U_trial
            V = V_trial
            W = W_trial
        return U, V, W

    def alternating(U, V, W, cycles=60):
        R = U.shape[1]
        for c in range(cycles):
            if time.time() > deadline - 4:
                break
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
        return U, V, W

    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def try_rank(R, seed=42, n_inits=12):
        if time.time() > deadline - 10:
            return None
        np.random.seed(seed)
        best = None
        best_err = float('inf')
        for init_iter in range(n_inits):
            if time.time() > deadline - 5:
                break
            # Mix of random and structured initializations
            if init_iter % 3 == 0:
                # Random
                U = np.random.randn(U_rows, R) * 0.5
                V = np.random.randn(V_rows, R) * 0.5
                W = np.random.randn(W_rows, R) * 0.5
            elif init_iter % 3 == 1:
                # Sparse random
                U = np.random.randn(U_rows, R)
                V = np.random.randn(V_rows, R)
                W = np.random.randn(W_rows, R)
                # Zero out ~50% of entries
                mask_U = np.random.rand(U_rows, R) < 0.5
                mask_V = np.random.rand(V_rows, R) < 0.5
                mask_W = np.random.rand(W_rows, R) < 0.5
                U[mask_U] = 0
                V[mask_V] = 0
                W[mask_W] = 0
                U *= 0.3
                V *= 0.3
                W *= 0.3
            else:
                # Block-structured
                U = np.zeros((U_rows, R))
                V = np.zeros((V_rows, R))
                W = np.zeros((W_rows, R))
                for r in range(R):
                    # Randomly assign some entries
                    U[np.random.choice(U_rows, size=max(1, U_rows//3), replace=False), r] = np.random.choice([-1, 1], size=max(1, U_rows//3))
                    V[np.random.choice(V_rows, size=max(1, V_rows//3), replace=False), r] = np.random.choice([-1, 1], size=max(1, V_rows//3))
                    W[np.random.choice(W_rows, size=max(1, W_rows//3), replace=False), r] = np.random.choice([-1, 1], size=max(1, W_rows//3))
            # Normalize columns
            for r in range(R):
                for M in (U, V, W):
                    mx = np.max(np.abs(M[:, r]))
                    if mx > 0:
                        M[:, r] /= mx
            U, V, W = alternating(U, V, W, cycles=40)
            U, V, W = gauss_newton(U, V, W, max_iter=300, tol=1e-10)
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-9:
                Us, Vs, Ws = snap_to_half_integer(U, V, W)
                err_snapped = max_abs_residual(Us, Vs, Ws)
                if err_snapped < 1e-7:
                    return Us, Vs, Ws
                else:
                    return U, V, W
        if best is not None and best_err < 1e-7:
            Us, Vs, Ws = snap_to_half_integer(best[0], best[1], best[2])
            err_snapped = max_abs_residual(Us, Vs, Ws)
            if err_snapped < 1e-7:
                return Us, Vs, Ws
            else:
                return best
        return None

    # Start with schoolbook
    best_rank = N
    U_best, V_best, W_best = build_schoolbook(best_rank)

    # Try ranks downward — include lower ranks that might be achievable
    for R in range(48, 28, -1):
        if time.time() > deadline - 15:
            break
        found = False
        for seed in range(15):
            if time.time() > deadline - 8:
                break
            result = try_rank(R, seed=seed*17 + 7)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            # Continue trying to go lower for a bit
            if time.time() > deadline - 20:
                break

    # Final high-precision polish
    if best_rank < N and time.time() < deadline - 4:
        U_best, V_best, W_best = gauss_newton(U_best, V_best, W_best, max_iter=500, tol=1e-13)
        # Try snapping after polish
        Us, Vs, Ws = snap_to_half_integer(U_best, V_best, W_best)
        err_snapped = max_abs_residual(Us, Vs, Ws)
        if err_snapped < 1e-7:
            U_best, V_best, W_best = Us, Vs, Ws

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
