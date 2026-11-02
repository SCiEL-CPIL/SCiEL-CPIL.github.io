"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 142.0  # safety margin under 150s wall-clock

    n, m, p = 2, 5, 5
    # Tensor convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)

    def build_schoolbook(rank):
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

    def compute_residual(U, V, W):
        # residual for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        R = U.shape[1]
        res = np.zeros(n*m*p)
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

    # Precompute index maps for fast linear solves
    # For U: variable U[i*m+j, r]. Constraint (i,j,k): sum_r U[i*m+j,r]*(V[j*p+k,r]*W[k*n+i,r]) = 1
    # For V: variable V[j*p+k, r]. Constraint (i,j,k): sum_r V[j*p+k,r]*(U[i*m+j,r]*W[k*n+i,r]) = 1
    # For W: variable W[k*n+i, r]. Constraint (i,j,k): sum_r W[k*n+i,r]*(U[i*m+j,r]*V[j*p+k,r]) = 1

    def solve_for_U(V, W, R):
        A = np.zeros((n*m*p, n*m*R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i*m*p + j*p + k
                    for r in range(R):
                        A[row, (i*m+j)*R + r] = V[j*p+k, r] * W[k*n+i, r]
        b = np.ones(n*m*p)
        try:
            u_flat = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            u_flat = np.zeros(n*m*R)
        return u_flat.reshape(n*m, R)

    def solve_for_V(U, W, R):
        A = np.zeros((n*m*p, m*p*R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i*m*p + j*p + k
                    for r in range(R):
                        A[row, (j*p+k)*R + r] = U[i*m+j, r] * W[k*n+i, r]
        b = np.ones(n*m*p)
        try:
            v_flat = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            v_flat = np.zeros(m*p*R)
        return v_flat.reshape(m*p, R)

    def solve_for_W(U, V, R):
        A = np.zeros((n*m*p, p*n*R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i*m*p + j*p + k
                    for r in range(R):
                        A[row, (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
        b = np.ones(n*m*p)
        try:
            w_flat = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            w_flat = np.zeros(p*n*R)
        return w_flat.reshape(p*n, R)

    def gauss_newton(U, V, W, max_iter=400, tol=1e-11):
        R = U.shape[1]
        total_vars = U.size + V.size + W.size
        for it in range(max_iter):
            if time.time() > deadline - 4:
                break
            res = compute_residual(U, V, W)
            if np.max(np.abs(res)) < tol:
                break
            J = np.zeros((n*m*p, total_vars))
            idx = 0
            for i in range(n):
                for j in range(m):
                    u_idx = i*m+j
                    for k in range(p):
                        v_idx = j*p+k
                        w_idx = k*n+i
                        for r in range(R):
                            J[idx, u_idx*R + r] = V[v_idx, r] * W[w_idx, r]
                            J[idx, U.size + v_idx*R + r] = U[u_idx, r] * W[w_idx, r]
                            J[idx, U.size + V.size + w_idx*R + r] = U[u_idx, r] * V[v_idx, r]
                        idx += 1
            try:
                JtJ = J.T @ J
                JtJ += 1e-9 * np.eye(total_vars)
                delta = np.linalg.solve(JtJ, -J.T @ res)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(JtJ, -J.T @ res, rcond=None)[0]
                except Exception:
                    break
            Uf = U.flatten() + delta[:U.size]
            Vf = V.flatten() + delta[U.size:U.size+V.size]
            Wf = W.flatten() + delta[U.size+V.size:]
            U = Uf.reshape(U.shape)
            V = Vf.reshape(V.shape)
            W = Wf.reshape(W.shape)
        return U, V, W

    def alternating(U, V, W, cycles=40):
        R = U.shape[1]
        for c in range(cycles):
            if time.time() > deadline - 6:
                break
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
        return U, V, W

    def try_rank(R, seed=42):
        if time.time() > deadline - 12:
            return None
        np.random.seed(seed)
        best = None
        best_err = float('inf')
        for init_iter in range(4):
            if time.time() > deadline - 6:
                break
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            # normalize columns to avoid degenerate scales
            for r in range(R):
                for M in (U, V, W):
                    mx = np.max(np.abs(M[:, r]))
                    if mx > 0:
                        M[:, r] /= mx
            U, V, W = alternating(U, V, W, cycles=30)
            U, V, W = gauss_newton(U, V, W, max_iter=150, tol=1e-10)
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-9:
                return U, V, W
        if best is not None and best_err < 1e-7:
            return best
        return None

    # Start with schoolbook
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)

    # Try ranks downward from a reasonable bound
    for R in range(45, 37, -1):
        if time.time() > deadline - 18:
            break
        found = False
        for seed in range(12):
            if time.time() > deadline - 10:
                break
            result = try_rank(R, seed=seed*13 + 3)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            break

    # Final polish
    if best_rank < n * m * p and time.time() < deadline - 5:
        U_best, V_best, W_best = gauss_newton(U_best, V_best, W_best, max_iter=300, tol=1e-12)

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
