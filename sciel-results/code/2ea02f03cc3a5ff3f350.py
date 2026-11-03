"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    n, m, p = 2, 5, 5
    N = n * m * p  # 50 total constraints

    # Precompute index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def build_schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_residual_fast(U, V, W):
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        prod = U_sel * V_sel * W_sel
        res = np.sum(prod, axis=1) - 1.0
        return res

    def max_abs_residual(U, V, W):
        return np.max(np.abs(compute_residual_fast(U, V, W)))

    def smooth_residual(U, V, W, beta=100.0):
        """Smooth surrogate for max-abs residual using log-sum-exp."""
        res = compute_residual_fast(U, V, W)
        abs_res = np.abs(res)
        # Soft-max via log-sum-exp for numerical stability
        max_val = np.max(abs_res)
        if max_val < 1e-30:
            return 0.0
        shifted = abs_res - max_val
        log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
        return log_sum_exp

    def smooth_residual_grad(U, V, W, beta=100.0):
        """Gradient of smooth residual objective."""
        res = compute_residual_fast(U, V, W)
        abs_res = np.abs(res)
        max_val = np.max(abs_res)
        if max_val < 1e-30:
            return None
        shifted = abs_res - max_val
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        
        sign_res = np.sign(res)
        weighted = weights * sign_res
        
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] * weighted
        for r in range(U.shape[1]):
            coeff = V_sel[:, r] * W_sel[:, r] * weighted
            np.add.at(U_grad, (u_idx, r), coeff)
            coeff = U_sel[:, r] * W_sel[:, r] * weighted
            np.add.at(V_grad, (v_idx, r), coeff)
            coeff = U_sel[:, r] * V_sel[:, r] * weighted
            np.add.at(W_grad, (w_idx, r), coeff)
        
        return U_grad, V_grad, W_grad

    def als_step(U, V, W, max_iter=80, tol=1e-12):
        R = U.shape[1]
        b = np.ones(N)

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Solve for U
            A_U = np.zeros((N, n * m * R))
            V_sel = V[v_idx, :]
            W_sel = W[w_idx, :]
            coeff = V_sel * W_sel
            for col in range(R):
                for u_row in range(n * m):
                    mask = (u_idx == u_row)
                    if np.any(mask):
                        A_U[mask, u_row * R + col] = coeff[mask, col]
            try:
                U_flat = np.linalg.lstsq(A_U, b, rcond=None)[0]
                U = U_flat.reshape(n * m, R)
            except:
                pass

            # Solve for V
            A_V = np.zeros((N, m * p * R))
            U_sel = U[u_idx, :]
            W_sel = W[w_idx, :]
            coeff = U_sel * W_sel
            for col in range(R):
                for v_row in range(m * p):
                    mask = (v_idx == v_row)
                    if np.any(mask):
                        A_V[mask, v_row * R + col] = coeff[mask, col]
            try:
                V_flat = np.linalg.lstsq(A_V, b, rcond=None)[0]
                V = V_flat.reshape(m * p, R)
            except:
                pass

            # Solve for W
            A_W = np.zeros((N, p * n * R))
            U_sel = U[u_idx, :]
            V_sel = V[v_idx, :]
            coeff = U_sel * V_sel
            for col in range(R):
                for w_row in range(p * n):
                    mask = (w_idx == w_row)
                    if np.any(mask):
                        A_W[mask, w_row * R + col] = coeff[mask, col]
            try:
                W_flat = np.linalg.lstsq(A_W, b, rcond=None)[0]
                W = W_flat.reshape(p * n, R)
            except:
                pass

            err = max_abs_residual(U, V, W)
            if err < tol:
                break

        return U, V, W

    def gauss_newton_fast(U, V, W, max_iter=40, tol=1e-14):
        R = U.shape[1]

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            res = compute_residual_fast(U, V, W)
            if np.max(np.abs(res)) < tol:
                break

            Np = n * m * R + m * p * R + p * n * R
            J = np.zeros((N, Np))
            
            U_sel = U[u_idx, :]
            V_sel = V[v_idx, :]
            W_sel = W[w_idx, :]
            
            off_u = 0
            off_v = n * m * R
            off_w = n * m * R + m * p * R
            
            for r in range(R):
                J[:, off_u + (u_idx * R + r)] = V_sel[:, r] * W_sel[:, r]
                J[:, off_v + (v_idx * R + r)] = U_sel[:, r] * W_sel[:, r]
                J[:, off_w + (w_idx * R + r)] = U_sel[:, r] * V_sel[:, r]

            try:
                JtJ = J.T @ J
                JtJ += 1e-12 * np.eye(Np)
                delta = np.linalg.solve(JtJ, -J.T @ res)
            except:
                break

            best_err = np.max(np.abs(res))
            step = 1.0
            for _ in range(10):
                U_t = U.copy(); V_t = V.copy(); W_t = W.copy()
                U_t += step * delta[:n * m * R].reshape(n * m, R)
                V_t += step * delta[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
                W_t += step * delta[n * m * R + m * p * R:].reshape(p * n, R)
                err_t = max_abs_residual(U_t, V_t, W_t)
                if err_t < best_err:
                    U, V, W = U_t, V_t, W_t
                    best_err = err_t
                    break
                step *= 0.5
            else:
                break

        return U, V, W

    def smooth_optimize(U, V, W, max_iter=50, tol=1e-12, beta=100.0):
        """Use smooth surrogate for optimization."""
        R = U.shape[1]
        
        def flatten_params(params):
            U_flat = params[:n * m * R]
            V_flat = params[n * m * R:n * m * R + m * p * R]
            W_flat = params[n * m * R + m * p * R:]
            return U_flat.reshape(n * m, R), V_flat.reshape(m * p, R), W_flat.reshape(p * n, R)
        
        def objective_and_grad(params):
            U, V, W = flatten_params(params)
            val = smooth_residual(U, V, W, beta)
            grads = smooth_residual_grad(U, V, W, beta)
            if grads is None:
                return val, np.zeros_like(params)
            U_g, V_g, W_g = grads
            grad = np.concatenate([U_g.flatten(), V_g.flatten(), W_g.flatten()])
            return val, grad
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            val, grad = objective_and_grad(params)
            if val < tol:
                break
            
            # Simple gradient descent with line search
            lr = 0.01
            best_val = val
            for ls in range(10):
                params_new = params - lr * grad
                U_new, V_new, W_new = flatten_params(params_new)
                new_val = smooth_residual(U_new, V_new, W_new, beta)
                if new_val < best_val:
                    params = params_new
                    best_val = new_val
                    break
                lr *= 0.5
            else:
                break
        
        U, V, W = flatten_params(params)
        return U, V, W

    def structured_init(R, seed):
        """Structured initialization exploiting 2x5x5 structure."""
        np.random.seed(seed)
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        strategy = seed % 5
        
        if strategy == 0:
            # Equal split between output rows
            R_per_row = R // n
            R_rem = R % n
            for i in range(n):
                start = i * R_per_row + min(i, R_rem)
                end = start + R_per_row + (1 if i < R_rem else 0)
                U[i * m:(i+1) * m, start:end] = np.random.randn(m, end - start) * 0.3
                V[:, start:end] = np.random.randn(m * p, end - start) * 0.3
                W[i::n, start:end] = np.random.randn(p, end - start) * 0.3
        elif strategy == 1:
            # Group by input columns
            R_per_col = R // m
            R_rem = R % m
            for j in range(m):
                start = j * R_per_col + min(j, R_rem)
                end = start + R_per_col + (1 if j < R_rem else 0)
                U[:, start:end] = np.random.randn(n * m, end - start) * 0.3
                V[j * p:(j+1) * p, start:end] = np.random.randn(p, end - start) * 0.3
                W[:, start:end] = np.random.randn(p * n, end - start) * 0.3
        elif strategy == 2:
            # Sparse random
            scale = 0.4
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            U[np.random.rand(*U.shape) < 0.3] = 0
            V[np.random.rand(*V.shape) < 0.3] = 0
            W[np.random.rand(*W.shape) < 0.3] = 0
        elif strategy == 3:
            # Strassen-like block structure
            # For 2x5x5, try to exploit the small n dimension
            R_block = max(1, R // 4)
            for b in range(4):
                start = b * R_block
                end = min(start + R_block, R)
                if start >= R:
                    break
                U[:, start:end] = np.random.randn(n * m, end - start) * 0.25
                V[:, start:end] = np.random.randn(m * p, end - start) * 0.25
                W[:, start:end] = np.random.randn(p * n, end - start) * 0.25
        else:
            # Mixed structure
            R_row = R // 3
            R_col = R // 3
            R_mix = R - 2 * R_row
            for i in range(n):
                start = i * (R_row // n) if n > 1 else 0
                end = start + R_row // n + (1 if i < R_row % n else 0)
                U[i * m:(i+1) * m, start:end] = np.random.randn(m, end - start) * 0.3
                V[:, start:end] = np.random.randn(m * p, end - start) * 0.3
                W[i::n, start:end] = np.random.randn(p, end - start) * 0.3
            col_start = R_row
            for j in range(m):
                start = col_start + j * (R_col // m)
                end = start + R_col // m + (1 if j < R_col % m else 0)
                U[:, start:end] = np.random.randn(n * m, end - start) * 0.3
                V[j * p:(j+1) * p, start:end] = np.random.randn(p, end - start) * 0.3
                W[:, start:end] = np.random.randn(p * n, end - start) * 0.3
            mix_start = R_row + R_col
            if mix_start < R:
                U[:, mix_start:R] = np.random.randn(n * m, R - mix_start) * 0.3
                V[:, mix_start:R] = np.random.randn(m * p, R - mix_start) * 0.3
                W[:, mix_start:R] = np.random.randn(p * n, R - mix_start) * 0.3
        
        return U, V, W

    def try_rank(R, seed=42):
        if time.time() > deadline - 15:
            return None

        best_result = None
        best_err = float('inf')
        
        num_inits = 10 if R <= 35 else 6

        for init_iter in range(num_inits):
            if time.time() > deadline - 8:
                break

            if init_iter == 0:
                U, V, W = structured_init(R, seed + init_iter * 100)
            elif init_iter == 1:
                np.random.seed(seed + init_iter * 7)
                U = np.random.randn(n * m, R) * 0.3
                V = np.random.randn(m * p, R) * 0.3
                W = np.random.randn(p * n, R) * 0.3
            elif init_iter == 2:
                np.random.seed(seed + init_iter * 11)
                U = np.random.randn(n * m, R) * 0.3
                V = np.random.randn(m * p, R) * 0.3
                W = np.random.randn(p * n, R) * 0.3
                U[np.random.rand(*U.shape) < 0.5] = 0
                V[np.random.rand(*V.shape) < 0.5] = 0
                W[np.random.rand(*W.shape) < 0.5] = 0
            elif init_iter == 3:
                np.random.seed(seed + init_iter * 13)
                scale = 0.15 + 0.1 * (init_iter % 4)
                U = np.random.randn(n * m, R) * scale
                V = np.random.randn(m * p, R) * scale
                W = np.random.randn(p * n, R) * scale
            else:
                np.random.seed(seed + init_iter * 17)
                U = np.random.randn(n * m, R) * 0.2
                V = np.random.randn(m * p, R) * 0.2
                W = np.random.randn(p * n, R) * 0.2

            # Phase 1: ALS
            U, V, W = als_step(U, V, W, max_iter=60, tol=1e-12)
            
            # Phase 2: Smooth optimization
            if time.time() < deadline - 10:
                U, V, W = smooth_optimize(U, V, W, max_iter=30, tol=1e-12, beta=100.0)
            
            # Phase 3: Gauss-Newton polish
            if time.time() < deadline - 8:
                U, V, W = gauss_newton_fast(U, V, W, max_iter=30, tol=1e-14)

            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())

            if err < 1e-10:
                return U, V, W

        if best_result is not None and best_err < 1e-8:
            return best_result
        return None

    # Start with schoolbook
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)

    # Try ranks from lower values - focus on promising range
    ranks_to_try = list(range(28, 42))
    
    for R in ranks_to_try:
        if time.time() > deadline - 25:
            break
        found = False
        num_seeds = 12 if R <= 35 else 8
        for seed in range(num_seeds):
            if time.time() > deadline - 12:
                break
            result = try_rank(R, seed=seed * 19 + 5)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            if time.time() < deadline - 35:
                continue
            else:
                break

    # Final high-precision polish
    if best_rank < n * m * p and time.time() < deadline - 12:
        U_best, V_best, W_best = als_step(U_best, V_best, W_best, max_iter=100, tol=1e-14)
        if time.time() < deadline - 8:
            U_best, V_best, W_best = smooth_optimize(U_best, V_best, W_best, max_iter=50, tol=1e-14, beta=200.0)
        if time.time() < deadline - 5:
            U_best, V_best, W_best = gauss_newton_fast(U_best, V_best, W_best, max_iter=50, tol=1e-15)

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
