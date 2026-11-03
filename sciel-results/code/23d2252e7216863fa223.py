import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Lower bound for rank is max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)

    # Precompute index mappings for efficient tensor evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)

    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx

    def compute_tensor(U, V, W, R):
        """Compute tensor values: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]"""
        tensor_vals = np.zeros(n * m * p)
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            tensor_vals += u_r[U_idx] * v_r[V_idx] * w_r[W_idx]
        return tensor_vals

    def tensor_error(U, V, W, R):
        """Compute squared error of tensor decomposition."""
        tensor_vals = compute_tensor(U, V, W, R)
        return np.sum((tensor_vals - 1.0) ** 2)

    def polish(U, V, W, R, max_iter=200):
        """High-precision polish using L-BFGS-B on all parameters."""
        from scipy.optimize import minimize

        def objective(params):
            U_flat = params[:n * m * R]
            V_flat = params[n * m * R : n * m * R + m * p * R]
            W_flat = params[n * m * R + m * p * R :]

            U_mat = U_flat.reshape(n * m, R)
            V_mat = V_flat.reshape(m * p, R)
            W_mat = W_flat.reshape(p * n, R)

            tensor_vals = np.zeros(n * m * p)
            for r in range(R):
                u_r = U_mat[:, r]
                v_r = V_mat[:, r]
                w_r = W_mat[:, r]
                tensor_vals += u_r[U_idx] * v_r[V_idx] * w_r[W_idx]

            return np.sum((tensor_vals - 1.0) ** 2)

        init_params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        result = minimize(objective, init_params, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})

        U_flat = result.x[:n * m * R]
        V_flat = result.x[n * m * R : n * m * R + m * p * R]
        W_flat = result.x[n * m * R + m * p * R :]

        U_pol = U_flat.reshape(n * m, R)
        V_pol = V_flat.reshape(m * p, R)
        W_pol = W_flat.reshape(p * n, R)

        return U_pol, V_pol, W_pol, result.fun

    def try_rank(R, seed_base=42):
        """Try to find a valid decomposition at rank R using ALS + polish."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None

        # Multiple random restarts
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            np.random.seed(seed_base + restart * 137)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5

            # ALS iterations
            max_als_iters = 100
            for it in range(max_als_iters):
                if time.time() > deadline:
                    break

                # Solve for U: For each (i,j), U[i*m+j, :] should satisfy
                # dot(U[i*m+j, :], A_u[i*m+j, :]) = 1
                # where A_u[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                A_u = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        W_idx_i = np.array([k * n + i for k in range(p)])
                        W_slice = W[W_idx_i, :]  # shape (p, R)
                        V_slice = V[j * p:(j + 1) * p, :]  # shape (p, R)
                        A_u[i * m + j, :] = np.sum(V_slice * W_slice, axis=0)

                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq == 0] = 1e-12
                U = A_u / norms_sq[:, np.newaxis]

                # Solve for V: For each (j,k), V[j*p+k, :] should satisfy
                # dot(V[j*p+k, :], A_v[j*p+k, :]) = 1
                # where A_v[j*p+k, r] = sum_i U[i*m+j, r] * W[k*n+i, r]
                A_v = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        U_idx_j = np.array([i * m + j for i in range(n)])
                        U_slice = U[U_idx_j, :]  # shape (n, R)
                        W_idx_k = np.array([k * n + i for i in range(n)])
                        W_slice = W[W_idx_k, :]  # shape (n, R)
                        A_v[j * p + k, :] = np.sum(U_slice * W_slice, axis=0)

                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v == 0] = 1e-12
                V = A_v / norms_sq_v[:, np.newaxis]

                # Solve for W: For each (k,i), W[k*n+i, :] should satisfy
                # dot(W[k*n+i, :], A_w[k*n+i, :]) = 1
                # where A_w[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                A_w = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        U_idx_i = np.array([i * m + j for j in range(m)])
                        U_slice = U[U_idx_i, :]  # shape (m, R)
                        V_idx_k = np.array([j * p + k for j in range(m)])
                        V_slice = V[V_idx_k, :]  # shape (m, R)
                        A_w[k * n + i, :] = np.sum(U_slice * V_slice, axis=0)

                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w == 0] = 1e-12
                W = A_w / norms_sq_w[:, np.newaxis]

                # Compute error
                err = tensor_error(U, V, W, R)
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if best_error < 1e-8:
                    break

            # If ALS converged well, do a polish pass
            if best_error < 1e-4 and time.time() < deadline:
                U_pol, V_pol, W_pol, polish_err = polish(best_U, best_V, best_W, R, max_iter=300)
                if polish_err < best_error:
                    best_error = polish_err
                    best_U, best_V, best_W = U_pol, V_pol, W_pol

            if best_error < 1e-6:
                break

        if best_error < 1e-5:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2

            # Verify exactness
            tensor_vals = compute_tensor(U_int, V_int, W_int, R)
            if np.all(np.abs(tensor_vals - 1.0) < 1e-9):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()

        return None

    # Try ranks from the lower bound upward
    for rank_offset in range(0, 15):
        if time.time() > deadline:
            break

        R = min_rank + rank_offset
        result = try_rank(R, seed_base=42 + rank_offset * 1000)

        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook if no low-rank solution found
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
