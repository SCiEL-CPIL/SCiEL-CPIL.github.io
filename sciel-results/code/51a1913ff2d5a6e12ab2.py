import time
import numpy as np
import random

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 4
    TIME_LIMIT = 120.0

    def time_up():
        return (time.time() - start_time) > TIME_LIMIT - 5.0

    def verify(U, V, W, rank):
        """Verify decomposition correctness."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])

        T_target = np.ones((n, m, p), dtype=float)
        return np.allclose(T_recon, T_target, atol=1e-9)

    def build_strassen():
        """Build the base rank-28 Strassen decomposition."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        strassen_steps = [
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
            {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
        ]

        def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
            for s in range(7):
                step = strassen_steps[s]
                col = col_offset + s

                for local_idx, coeff in step['u'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = l_row
                    global_j = a_col_start + l_col
                    global_idx = global_i * m + global_j
                    U[global_idx][col] += coeff

                for local_idx, coeff in step['v'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = b_row_start + l_row
                    global_j = b_col_start + l_col
                    global_idx = global_i * p + global_j
                    V[global_idx][col] += coeff

                for local_idx, coeff in step['w'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_k = c_col_start + l_col
                    global_i = l_row
                    global_idx = global_k * n + global_i
                    W[global_idx][col] += coeff

        add_block(0, 0, 0, 0, 0)
        add_block(0, 0, 2, 2, 7)
        add_block(2, 2, 0, 0, 14)
        add_block(2, 2, 2, 2, 21)

        return U, V, W, rank

    def try_reduce_rank(U, V, W, current_rank, target_rank):
        """
        Try to reduce rank by finding linear dependencies.
        """
        if current_rank <= target_rank:
            return None

        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for elim_idx in range(current_rank - 1, target_rank - 1, -1):
            if time_up():
                break

            other_indices = [j for j in range(current_rank) if j != elim_idx]
            num_unknowns = len(other_indices)

            U_target = U_arr[:, elim_idx]
            U_matrix = np.column_stack([U_arr[:, j] for j in other_indices])

            V_target = V_arr[:, elim_idx]
            V_matrix = np.column_stack([V_arr[:, j] for j in other_indices])

            W_target = W_arr[:, elim_idx]
            W_matrix = np.column_stack([W_arr[:, j] for j in other_indices])

            A_combined = np.vstack([U_matrix, V_matrix, W_matrix])
            b_combined = np.concatenate([U_target, V_target, W_target])

            try:
                coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_combined, b_combined, rcond=None)

                residual = np.linalg.norm(A_combined @ coeffs - b_combined)
                if residual < 1e-9:
                    new_U = [U[i][:] for i in range(len(U))]
                    new_V = [V[i][:] for i in range(len(V))]
                    new_W = [W[i][:] for i in range(len(W))]

                    for new_idx in range(len(other_indices)):
                        j = other_indices[new_idx]
                        c = coeffs[new_idx]
                        if abs(c) > 1e-12:
                            for i in range(len(U)):
                                new_U[i][j] += c * U[i][elim_idx]
                                new_V[i][j] += c * V[i][elim_idx]
                                new_W[i][j] += c * W[i][elim_idx]

                    new_U = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_U]
                    new_V = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_V]
                    new_W = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_W]

                    return new_U, new_V, new_W, current_rank - 1
            except:
                pass

        return None

    def try_tensor_flattening(U, V, W, current_rank, target_rank):
        """
        Try to reduce rank using tensor flattening approach.
        Reshape the multiplication tensor and find a sparse decomposition.
        """
        if current_rank <= target_rank:
            return None

        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Build the target tensor
        T_target = np.ones((n, m, p), dtype=float)

        # Try to find a rank-reduced decomposition by solving a linear system
        # We want to find U', V', W' of rank target_rank such that
        # sum_{r=1}^{target_rank} U'_r ⊗ V'_r ⊗ W'_r = T_target

        # This is a non-convex problem, so we use a heuristic approach
        # Try random initializations and use gradient descent

        best_result = None
        best_residual = float('inf')

        for attempt in range(5):
            if time_up():
                break

            # Random initialization
            np.random.seed(attempt * 12345)
            U_try = np.random.randn(n * m, target_rank) * 0.1
            V_try = np.random.randn(m * p, target_rank) * 0.1
            W_try = np.random.randn(p * n, target_rank) * 0.1

            # Simple gradient descent
            lr = 0.01
            for step in range(100):
                if time_up():
                    break

                # Compute current tensor
                T_current = np.zeros((n, m, p), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_current[i, j, k] = np.sum(U_try[i*m+j, :] * V_try[j*p+k, :] * W_try[k*n+i, :])

                # Compute gradient
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = T_current[i, j, k] - T_target[i, j, k]
                            for r in range(target_rank):
                                grad_U[i*m+j, r] += diff * V_try[j*p+k, r] * W_try[k*n+i, r]
                                grad_V[j*p+k, r] += diff * U_try[i*m+j, r] * W_try[k*n+i, r]
                                grad_W[k*n+i, r] += diff * U_try[i*m+j, r] * V_try[j*p+k, r]

                # Update
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W

            # Check final residual
            T_final = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_final[i, j, k] = np.sum(U_try[i*m+j, :] * V_try[j*p+k, :] * W_try[k*n+i, :])

            residual = np.linalg.norm(T_final - T_target)
            if residual < best_residual:
                best_residual = residual
                best_result = (U_try, V_try, W_try)

        if best_result is not None and best_residual < 1e-6:
            U_try, V_try, W_try = best_result
            return (U_try.tolist(), V_try.tolist(), W_try.tolist(), target_rank)

        return None

    # Start with Strassen
    U, V, W, rank = build_strassen()

    # Verify base case
    if not verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W, best_rank = U, V, W, rank

    # Try to reduce rank using linear dependency search
    target_rank = 24

    while rank > target_rank and not time_up():
        result = try_reduce_rank(U, V, W, rank, target_rank)
        if result is None:
            break

        U, V, W, rank = result

        if verify(U, V, W, rank):
            best_U, best_V, best_W, best_rank = U, V, W, rank
        else:
            U, V, W, rank = best_U, best_V, best_W, best_rank
            break

    # Try tensor flattening approach
    if best_rank > 24 and not time_up():
        result = try_tensor_flattening(U, V, W, best_rank, 24)
        if result is not None:
            U_try, V_try, W_try, rank_try = result
            if verify(U_try, V_try, W_try, rank_try):
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, rank_try

    return {"U": best_U, "V": best_V, "W": best_W}
