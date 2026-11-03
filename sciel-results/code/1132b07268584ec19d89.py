import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    N = n * m
    M = m * p
    P = p * n
    T_size = N * M * P

    # Precompute index mappings for tensor evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)

    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx

    def compute_error(U, V, W, R):
        """
        Compute squared error of tensor decomposition.
        """
        err = 0.0
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            vals = u_r[U_idx] * v_r[V_idx] * w_r[W_idx]
            err += np.sum((vals - 1.0) ** 2)
        return err

    def polish(U, V, W, R, max_iter=300):
        """
        Polish using L-BFGS-B.
        """
        def objective(params):
            U_flat = params[:N * R]
            V_flat = params[N * R : N * R + M * R]
            W_flat = params[N * R + M * R :]

            U_mat = U_flat.reshape(N, R)
            V_mat = V_flat.reshape(M, R)
            W_mat = W_flat.reshape(P, R)

            return compute_error(U_mat, V_mat, W_mat, R)

        init_params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        
        result = minimize(objective, init_params, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        
        U_flat = result.x[:N * R]
        V_flat = result.x[N * R : N * R + M * R]
        W_flat = result.x[N * R + M * R :]

        U_pol = U_flat.reshape(N, R)
        V_pol = V_flat.reshape(M, R)
        W_pol = W_flat.reshape(P, R)

        return U_pol, V_pol, W_pol, result.fun

    def try_rank(R, seed_base=42):
        """
        Try to find a valid decomposition at rank R.
        """
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None

        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            np.random.seed(seed_base + restart * 101)
            U = np.random.randn(N, R) * 0.1
            V = np.random.randn(M, R) * 0.1
            W = np.random.randn(P, R) * 0.1

            # ALS Phase
            max_als_iters = 300
            for it in range(max_als_iters):
                if time.time() > deadline:
                    break

                # Update U
                A_u = np.zeros((N, R))
                for i in range(n):
                    for j in range(m):
                        V_slice = V[j*p : (j+1)*p, :]
                        W_idx_i = np.arange(i, P, n)
                        W_slice = W[W_idx_i, :]
                        A_u[i*m+j, :] = np.sum(V_slice * W_slice, axis=0)
                
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq < 1e-12] = 1e-12
                U = A_u / norms_sq[:, np.newaxis]

                # Update V
                A_v = np.zeros((M, R))
                for j in range(m):
                    for k in range(p):
                        U_idx_j = np.arange(j, N, m)
                        U_slice = U[U_idx_j, :]
                        W_idx_k = np.arange(k, P, n)
                        W_slice = W[W_idx_k, :]
                        A_v[j*p+k, :] = np.sum(U_slice * W_slice, axis=0)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v < 1e-12] = 1e-12
                V = A_v / norms_sq_v[:, np.newaxis]

                # Update W
                A_w = np.zeros((P, R))
                for k in range(p):
                    for i in range(n):
                        U_idx_i = np.arange(i, N, m)
                        U_slice = U[U_idx_i, :]
                        V_idx_k = np.arange(k, M, p)
                        V_slice = V[V_idx_k, :]
                        A_w[k*n+i, :] = np.sum(U_slice * V_slice, axis=0)
                
                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w < 1e-12] = 1e-12
                W = A_w / norms_sq_w[:, np.newaxis]

                # Compute error
                err = compute_error(U, V, W, R)
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if best_error < 1e-7:
                    break

            # Polish Phase
            if best_error < 1e-4 and time.time() < deadline:
                U_pol, V_pol, W_pol, polish_err = polish(best_U, best_V, best_W, R, max_iter=300)
                if polish_err < best_error:
                    best_error = polish_err
                    best_U, best_V, best_W = U_pol, V_pol, W_pol

            if best_error < 1e-8:
                break

        if best_error < 1e-6:
            # Try rounding to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            err_int = compute_error(U_int, V_int, W_int, R)
            if err_int < 1e-9:
                return U_int, V_int, W_int
            
            # Try rounding to integers
            U_int2 = np.round(best_U)
            V_int2 = np.round(best_V)
            W_int2 = np.round(best_W)
            
            err_int2 = compute_error(U_int2, V_int2, W_int2, R)
            if err_int2 < 1e-9:
                return U_int2, V_int2, W_int2

        return None

    min_rank = max(N, M, P)
    
    # Try ranks from lower bound up to 95
    # If we don't find one, we return schoolbook (100)
    for R in range(min_rank, 96):
        if time.time() > deadline:
            break
            
        result = try_rank(R, seed_base=42 + R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
