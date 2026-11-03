"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin: 150s kill, leave 2s for return

    n, m, p = 2, 5, 5
    N = n * m * p  # 50 total constraints

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
        """Vectorized residual computation using einsum."""
        # Residual for (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
        # Reshape for broadcasting
        U_r = U[:, :, np.newaxis]  # (n*m, R, 1)
        V_r = V[:, :, np.newaxis]  # (m*p, R, 1)  
        W_r = W[:, :, np.newaxis]  # (p*n, R, 1)
        
        # For each (i,j,k), we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * p)  # (N,)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)  # (N,)
        k_idx = np.tile(np.arange(p), n * m)  # (N,)
        
        u_idx = i_idx * m + j_idx  # (N,)
        v_idx = j_idx * p + k_idx  # (N,)
        w_idx = k_idx * n + i_idx  # (N,)
        
        # Gather and compute
        U_sel = U[u_idx, :]  # (N, R)
        V_sel = V[v_idx, :]  # (N, R)
        W_sel = W[w_idx, :]  # (N, R)
        
        # Element-wise product and sum over R
        prod = U_sel * V_sel * W_sel  # (N, R)
        res = np.sum(prod, axis=1) - 1.0  # (N,)
        return res

    def max_abs_residual(U, V, W):
        return np.max(np.abs(compute_residual_fast(U, V, W)))

    def _build_A_U_fast(V, W, R):
        """Build linear system for U using vectorization."""
        A = np.zeros((N, n * m * R))
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx  # (N,)
        v_idx = j_idx * p + k_idx  # (N,)
        w_idx = k_idx * n + i_idx  # (N,)
        
        V_sel = V[v_idx, :]  # (N, R)
        W_sel = W[w_idx, :]  # (N, R)
        coeff = V_sel * W_sel  # (N, R)
        
        for col in range(R):
            row_start = 0
            for u_row in range(n * m):
                mask = (u_idx == u_row)
                if np.any(mask):
                    A[mask, u_row * R + col] = coeff[mask, col]
        return A

    def _build_A_V_fast(U, W, R):
        """Build linear system for V using vectorization."""
        A = np.zeros((N, m * p * R))
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx  # (N,)
        v_idx = j_idx * p + k_idx  # (N,)
        w_idx = k_idx * n + i_idx  # (N,)
        
        U_sel = U[u_idx, :]  # (N, R)
        W_sel = W[w_idx, :]  # (N, R)
        coeff = U_sel * W_sel  # (N, R)
        
        for col in range(R):
            for v_row in range(m * p):
                mask = (v_idx == v_row)
                if np.any(mask):
                    A[mask, v_row * R + col] = coeff[mask, col]
        return A

    def _build_A_W_fast(U, V, R):
        """Build linear system for W using vectorization."""
        A = np.zeros((N, p * n * R))
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx  # (N,)
        v_idx = j_idx * p + k_idx  # (N,)
        w_idx = k_idx * n + i_idx  # (N,)
        
        U_sel = U[u_idx, :]  # (N, R)
        V_sel = V[v_idx, :]  # (N, R)
        coeff = U_sel * V_sel  # (N, R)
        
        for col in range(R):
            for w_row in range(p * n):
                mask = (w_idx == w_row)
                if np.any(mask):
                    A[mask, w_row * R + col] = coeff[mask, col]
        return A

    def als_step(U, V, W, max_iter=100, tol=1e-10):
        R = U.shape[1]
        b = np.ones(N)

        for it in range(max_iter):
            if time.time() > deadline - 3:
                break

            A_U = _build_A_U_fast(V, W, R)
            try:
                U_flat = np.linalg.lstsq(A_U, b, rcond=None)[0]
                U = U_flat.reshape(n * m, R)
            except:
                pass

            A_V = _build_A_V_fast(U, W, R)
            try:
                V_flat = np.linalg.lstsq(A_V, b, rcond=None)[0]
                V = V_flat.reshape(m * p, R)
            except:
                pass

            A_W = _build_A_W_fast(U, V, R)
            try:
                W_flat = np.linalg.lstsq(A_W, b, rcond=None)[0]
                W = W_flat.reshape(p * n, R)
            except:
                pass

            err = max_abs_residual(U, V, W)
            if err < tol:
                break

        return U, V, W

    def gauss_newton_fast(U, V, W, max_iter=30, tol=1e-12):
        R = U.shape[1]

        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            res = compute_residual_fast(U, V, W)
            if np.max(np.abs(res)) < tol:
                break

            Np = n * m * R + m * p * R + p * n * R
            J = np.zeros((N, Np))
            
            i_idx = np.repeat(np.arange(n), m * p)
            j_idx = np.tile(np.repeat(np.arange(m), p), n)
            k_idx = np.tile(np.arange(p), n * m)
            
            u_idx = i_idx * m + j_idx
            v_idx = j_idx * p + k_idx
            w_idx = k_idx * n + i_idx
            
            U_sel = U[u_idx, :]  # (N, R)
            V_sel = V[v_idx, :]  # (N, R)
            W_sel = W[w_idx, :]  # (N, R)
            
            # Build Jacobian columns
            off_u = 0
            off_v = n * m * R
            off_w = n * m * R + m * p * R
            
            for r in range(R):
                # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                J[:, off_u + (u_idx * R + r)] = V_sel[:, r] * W_sel[:, r]
                # d/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                J[:, off_v + (v_idx * R + r)] = U_sel[:, r] * W_sel[:, r]
                # d/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                J[:, off_w + (w_idx * R + r)] = U_sel[:, r] * V_sel[:, r]

            try:
                JtJ = J.T @ J
                JtJ += 1e-10 * np.eye(Np)
                delta = np.linalg.solve(JtJ, -J.T @ res)
            except:
                break

            best_err = np.max(np.abs(res))
            step = 1.0
            for _ in range(8):
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

    def structured_init(R, seed):
        """Initialize using row-decomposition structure."""
        np.random.seed(seed)
        
        # For 2x5x5, exploit that we have 2 output rows
        # Initialize U with structure that helps row separation
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Split rank between the two rows
        R_per_row = R // n
        R_rem = R % n
        
        for i in range(n):
            start = i * R_per_row + min(i, R_rem)
            end = start + R_per_row + (1 if i < R_rem else 0)
            
            # For each row, initialize with random values
            U[i * m:(i+1) * m, start:end] = np.random.randn(m, end - start) * 0.5
            V[:, start:end] = np.random.randn(m * p, end - start) * 0.5
            W[i::n, start:end] = np.random.randn(p, end - start) * 0.5
        
        return U, V, W

    def try_rank(R, seed=42):
        if time.time() > deadline - 5:
            return None

        best_result = None
        best_err = float('inf')
        
        num_inits = 6 if R <= 32 else 4

        for init_iter in range(num_inits):
            if time.time() > deadline - 3:
                break

            if init_iter == 0:
                # Structured initialization
                U, V, W = structured_init(R, seed + init_iter * 100)
            elif init_iter == 1:
                # Pure random
                np.random.seed(seed + init_iter * 7)
                U = np.random.randn(n * m, R) * 0.4
                V = np.random.randn(m * p, R) * 0.4
                W = np.random.randn(p * n, R) * 0.4
            else:
                # Random with different scales
                np.random.seed(seed + init_iter * 13)
                scale = 0.2 + 0.1 * (init_iter % 3)
                U = np.random.randn(n * m, R) * scale
                V = np.random.randn(m * p, R) * scale
                W = np.random.randn(p * n, R) * scale

            # ALS refinement
            U, V, W = als_step(U, V, W, max_iter=40, tol=1e-11)
            
            # Gauss-Newton polish
            U, V, W = gauss_newton_fast(U, V, W, max_iter=15, tol=1e-12)

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

    # Try ranks from lower values - known results suggest rank 30-35 for 2x5x5
    ranks_to_try = list(range(30, 40))
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        found = False
        # More seeds for lower ranks (harder to find)
        num_seeds = 20 if R <= 32 else 12
        for seed in range(num_seeds):
            if time.time() > deadline - 8:
                break
            result = try_rank(R, seed=seed * 17 + 3)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            # Try to find even lower rank with remaining time
            if time.time() < deadline - 30:
                continue
            else:
                break

    # Final high-precision polish
    if best_rank < n * m * p and time.time() < deadline - 5:
        U_best, V_best, W_best = als_step(U_best, V_best, W_best, max_iter=80, tol=1e-13)
        U_best, V_best, W_best = gauss_newton_fast(U_best, V_best, W_best, max_iter=30, tol=1e-14)

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
