"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np
from fractions import Fraction
from itertools import product

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

    # Precompute index arrays for vectorized operations
    ijk_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ijk_indices.append((i*m+j, j*p+k, k*n+i))
    u_idx = np.array([x[0] for x in ijk_indices])  # (nmp,)
    v_idx = np.array([x[1] for x in ijk_indices])  # (nmp,)
    w_idx = np.array([x[2] for x in ijk_indices])  # (nmp,)
    nmp = n*m*p

    def compute_residual(U, V, W):
        prod = V[v_idx] * W[w_idx]  # (nmp, R)
        res = np.sum(U[u_idx] * prod, axis=1) - 1.0
        return res

    def max_abs_residual(U, V, W):
        return float(np.max(np.abs(compute_residual(U, V, W))))

    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
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
        V = np.zeros((m*p, R))
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
        W = np.zeros((p*n, R))
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

    def gauss_newton(U, V, W, max_iter=400, tol=1e-11):
        R = U.shape[1]
        total_vars = U.size + V.size + W.size
        U_flat = U.flatten().copy()
        V_flat = V.flatten().copy()
        W_flat = W.flatten().copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 4:
                break
            U_cur = U_flat.reshape(U.shape)
            V_cur = V_flat.reshape(V.shape)
            W_cur = W_flat.reshape(W.shape)
            res = compute_residual(U_cur, V_cur, W_cur)
            max_res = np.max(np.abs(res))
            if max_res < tol:
                break
            
            U_sel = U_cur[u_idx]  # (nmp, R)
            V_sel = V_cur[v_idx]  # (nmp, R)
            W_sel = W_cur[w_idx]  # (nmp, R)
            
            J_U = V_sel * W_sel  # (nmp, R)
            J_V = U_sel * W_sel  # (nmp, R)
            J_W = U_sel * V_sel  # (nmp, R)
            
            J = np.zeros((nmp, total_vars))
            for r in range(R):
                J[:, u_idx*R + r] = J_U[:, r]
                J[:, U.size + v_idx*R + r] = J_V[:, r]
                J[:, U.size + V.size + w_idx*R + r] = J_W[:, r]
            
            try:
                JtJ = J.T @ J
                JtJ += 1e-8 * np.eye(total_vars)
                rhs = -J.T @ res
                delta = np.linalg.solve(JtJ, rhs)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(JtJ, -J.T @ res, rcond=None)[0]
                except Exception:
                    break
            
            best_step = 1.0
            best_obj = np.sum(res**2)
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
                U_test = U_flat + alpha * delta[:U.size]
                V_test = V_flat + alpha * delta[U.size:U.size+V.size]
                W_test = W_flat + alpha * delta[U.size+V.size:]
                U_t = U_test.reshape(U.shape)
                V_t = V_test.reshape(V.shape)
                W_t = W_test.reshape(W.shape)
                res_t = compute_residual(U_t, V_t, W_t)
                obj_t = np.sum(res_t**2)
                if obj_t < best_obj:
                    best_obj = obj_t
                    best_step = alpha
            
            U_flat = U_flat + best_step * delta[:U.size]
            V_flat = V_flat + best_step * delta[U.size:U.size+V.size]
            W_flat = W_flat + best_step * delta[U.size+V.size:]
        return U_flat.reshape(U.shape), V_flat.reshape(V.shape), W_flat.reshape(W.shape)

    def alternating(U, V, W, cycles=40):
        R = U.shape[1]
        for c in range(cycles):
            if time.time() > deadline - 6:
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

    def snap_to_quarter_integer(U, V, W):
        U_s = np.round(U * 4) / 4
        V_s = np.round(V * 4) / 4
        W_s = np.round(W * 4) / 4
        return U_s, V_s, W_s

    def verify_exact(U, V, W, tol=1e-10):
        """Check if decomposition is exact within tolerance."""
        res = compute_residual(U, V, W)
        return np.max(np.abs(res)) < tol

    def snap_and_verify(U, V, W, max_denom=8):
        """Try to snap to rational values with small denominators and verify."""
        for denom in [2, 4, 8]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if verify_exact(U_s, V_s, W_s, tol=1e-9):
                return U_s, V_s, W_s
        U_s = np.round(U * 4) / 4
        V_s = np.round(V * 4) / 4
        W_s = np.round(W * 4) / 4
        if verify_exact(U_s, V_s, W_s, tol=1e-9):
            return U_s, V_s, W_s
        return None

    def tensor_completion(U, V, W, R, max_iter=100, tol=1e-12):
        """Tensor completion: project residual onto rank-R subspace via ALS."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            res = compute_residual(U, V, W)
            max_res = np.max(np.abs(res))
            if max_res < tol:
                break
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
        return U, V, W

    def random_rational_init(R, seed):
        """Initialize with random rational entries (small numerators/denominators)."""
        rng = np.random.RandomState(seed)
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for idx in range(n*m):
            for r in range(R):
                U[idx, r] = vals[rng.randint(len(vals))]
        for idx in range(m*p):
            for r in range(R):
                V[idx, r] = vals[rng.randint(len(vals))]
        for idx in range(p*n):
            for r in range(R):
                W[idx, r] = vals[rng.randint(len(vals))]
        return U, V, W

    def try_rank(R, seed=42):
        if time.time() > deadline - 12:
            return None
        np.random.seed(seed)
        best = None
        best_err = float('inf')
        num_inits = 20
        for init_iter in range(num_inits):
            if time.time() > deadline - 6:
                break
            if init_iter % 4 == 0:
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
            elif init_iter % 4 == 1:
                U = np.random.randn(n*m, R) * 0.3
                V = np.random.randn(m*p, R) * 0.3
                W = np.random.randn(p*n, R) * 0.3
                for r in range(min(3, R)):
                    U[:, r] += 0.5
                    V[:, r] += 0.5
                    W[:, r] += 0.5
            elif init_iter % 4 == 2:
                U = (np.random.rand(n*m, R) > 0.5).astype(float) * 0.8
                V = (np.random.rand(m*p, R) > 0.5).astype(float) * 0.8
                W = (np.random.rand(p*n, R) > 0.5).astype(float) * 0.8
            else:
                U, V, W = random_rational_init(R, seed + init_iter * 1000)
            
            for r in range(R):
                for M in (U, V, W):
                    mx = np.max(np.abs(M[:, r]))
                    if mx > 0:
                        M[:, r] /= mx
            U, V, W = tensor_completion(U, V, W, R, max_iter=60, tol=1e-11)
            U, V, W = gauss_newton(U, V, W, max_iter=200, tol=1e-11)
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-9:
                snapped = snap_and_verify(U, V, W)
                if snapped is not None:
                    return snapped
                else:
                    return U, V, W
        if best is not None and best_err < 1e-7:
            snapped = snap_and_verify(best[0], best[1], best[2])
            if snapped is not None:
                return snapped
            else:
                return best
        return None

    # Start with schoolbook
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)

    # Try ranks downward from a reasonable bound
    for R in range(42, 24, -1):
        if time.time() > deadline - 25:
            break
        found = False
        num_seeds = 12
        for seed in range(num_seeds):
            if time.time() > deadline - 15:
                break
            result = try_rank(R, seed=seed*13 + 7)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            continue
    
    # Final polish with tighter tolerance
    if best_rank < n * m * p and time.time() < deadline - 5:
        U_best, V_best, W_best = gauss_newton(U_best, V_best, W_best, max_iter=400, tol=1e-13)
        snapped = snap_and_verify(U_best, V_best, W_best)
        if snapped is not None:
            U_best, V_best, W_best = snapped
        else:
            Us, Vs, Ws = snap_to_half_integer(U_best, V_best, W_best)
            err_snapped = max_abs_residual(Us, Vs, Ws)
            if err_snapped < 1e-6:
                U_best, V_best, W_best = Us, Vs, Ws

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
