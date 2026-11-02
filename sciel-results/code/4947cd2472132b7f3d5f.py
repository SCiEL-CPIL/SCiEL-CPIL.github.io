import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 3, 4, 4

    # Build the local 3x2x2 tensor
    # T[a_idx, b_idx, c_idx] where:
    # a_idx = a_row * 2 + a_col (0..5)
    # b_idx = b_row * 2 + b_col (0..3)  
    # c_idx = c_row * 2 + c_col (0..5)
    # T = 1 if c_row == a_row and a_col == b_row and c_col == b_col
    T_local = np.zeros((6, 4, 6))
    for a_row in range(3):
        for a_col in range(2):
            for b_col in range(2):
                # C[a_row, b_col] += A[a_row, a_col] * B[a_col, b_col]
                T_local[a_row * 2 + a_col, a_col * 2 + b_col, a_row * 2 + b_col] = 1.0

    def try_find_decomposition(target_rank, max_time=25.0):
        """Try to find a rank-`target_rank` decomposition of T_local."""
        np.random.seed(42)
        best_err = float('inf')
        best_params = None
        best_U = best_V = best_W = None

        def compute_error(params):
            U_p = params[:6 * target_rank].reshape(6, target_rank)
            V_p = params[6 * target_rank:6 * target_rank + 4 * target_rank].reshape(4, target_rank)
            W_p = params[6 * target_rank + 4 * target_rank:].reshape(6, target_rank)
            T_recon = np.zeros((6, 4, 6))
            for r in range(target_rank):
                T_recon += np.einsum('i,j,k->ijk', U_p[:, r], V_p[:, r], W_p[:, r])
            return np.sum((T_local - T_recon) ** 2)

        def compute_error_fast(params):
            U_p = params[:6 * target_rank].reshape(6, target_rank)
            V_p = params[6 * target_rank:6 * target_rank + 4 * target_rank].reshape(4, target_rank)
            W_p = params[6 * target_rank + 4 * target_rank:].reshape(6, target_rank)
            # Use einsum for speed
            T_recon = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
            return np.sum((T_local - T_recon) ** 2)

        # Multiple random restarts
        for restart in range(10):
            if time.time() > start_time + deadline - 10:
                break

            # Random initialization
            U_init = np.random.randn(6, target_rank) * 0.5
            V_init = np.random.randn(4, target_rank) * 0.5
            W_init = np.random.randn(6, target_rank) * 0.5

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                result = minimize(
                    compute_error_fast,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_err:
                    best_err = result.fun
                    best_params = result.x
                    best_U = result.x[:6 * target_rank].reshape(6, target_rank)
                    best_V = result.x[6 * target_rank:6 * target_rank + 4 * target_rank].reshape(4, target_rank)
                    best_W = result.x[6 * target_rank + 4 * target_rank:].reshape(6, target_rank)

                    if best_err < 1e-8:
                        break
            except Exception:
                continue

            # Also try Nelder-Mead for short runs
            if time.time() < start_time + deadline - 15:
                try:
                    result_nm = minimize(
                        compute_error_fast,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if result_nm.fun < best_err:
                        best_err = result_nm.fun
                        best_params = result_nm.x
                        best_U = result_nm.x[:6 * target_rank].reshape(6, target_rank)
                        best_V = result_nm.x[6 * target_rank:6 * target_rank + 4 * target_rank].reshape(4, target_rank)
                        best_W = result_nm.x[6 * target_rank + 4 * target_rank:].reshape(6, target_rank)

                        if best_err < 1e-8:
                            break
                except Exception:
                    continue

        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err

    # Try to find rank 10 first, then rank 11 as fallback
    U_loc = V_loc = W_loc = None
    local_rank = 11

    if time.time() < start_time + deadline - 30:
        U10, V10, W10, err10 = try_find_decomposition(10, max_time=20.0)
        if U10 is not None and err10 < 1e-6:
            U_loc, V_loc, W_loc = U10, V10, W10
            local_rank = 10

    # If rank 10 failed, use the known rank-11 decomposition
    if U_loc is None:
        local_rank = 11
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))

        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1

        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1

        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1

        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1

        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1

        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1

        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1

        # Extra part (cols 7-10) for Row 2 of A
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1

        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1

        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1

        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1

    rank = 4 * local_rank

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * local_rank

        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset + local_rank] += U_loc[local_a_idx, :]

        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset + local_rank] += V_loc[local_b_idx, :]

        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset + local_rank] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)

    # Round to half-integers if possible
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    # Verify the decomposition
    T_global = np.zeros((n * m, m * p, p * n))
    for a_row in range(n):
        for a_col in range(m):
            for b_col in range(p):
                # C[a_row, b_col] += A[a_row, a_col] * B[a_col, b_col]
                T_global[a_row * m + a_col, a_col * p + b_col, b_col * n + a_row] = 1.0

    T_recon = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])

    if not np.allclose(T_global, T_recon, atol=1e-6):
        # If verification fails, fall back to the known rank-44 solution
        local_rank = 11
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        U_loc_fb = np.zeros((6, 11))
        V_loc_fb = np.zeros((4, 11))
        W_loc_fb = np.zeros((6, 11))

        U_loc_fb[0, 0] = 1; U_loc_fb[3, 0] = 1
        V_loc_fb[0, 0] = 1; V_loc_fb[3, 0] = 1
        W_loc_fb[0, 0] = 1; W_loc_fb[3, 0] = 1

        U_loc_fb[2, 1] = 1; U_loc_fb[3, 1] = 1
        V_loc_fb[0, 1] = 1
        W_loc_fb[2, 1] = 1; W_loc_fb[3, 1] = -1

        U_loc_fb[0, 2] = 1
        V_loc_fb[1, 2] = 1; V_loc_fb[3, 2] = -1
        W_loc_fb[1, 2] = 1; W_loc_fb[3, 2] = 1

        U_loc_fb[3, 3] = 1
        V_loc_fb[2, 3] = 1; V_loc_fb[0, 3] = -1
        W_loc_fb[0, 3] = 1; W_loc_fb[2, 3] = 1

        U_loc_fb[0, 4] = 1; U_loc_fb[1, 4] = 1
        V_loc_fb[3, 4] = 1
        W_loc_fb[0, 4] = -1; W_loc_fb[1, 4] = 1

        U_loc_fb[2, 5] = 1; U_loc_fb[0, 5] = -1
        V_loc_fb[0, 5] = 1; V_loc_fb[1, 5] = 1
        W_loc_fb[3, 5] = 1

        U_loc_fb[1, 6] = 1; U_loc_fb[3, 6] = -1
        V_loc_fb[2, 6] = 1; V_loc_fb[3, 6] = 1
        W_loc_fb[0, 6] = 1

        U_loc_fb[4, 7] = 1
        V_loc_fb[0, 7] = 1
        W_loc_fb[4, 7] = 1

        U_loc_fb[5, 8] = 1
        V_loc_fb[2, 8] = 1
        W_loc_fb[4, 8] = 1

        U_loc_fb[4, 9] = 1
        V_loc_fb[1, 9] = 1
        W_loc_fb[5, 9] = 1

        U_loc_fb[5, 10] = 1
        V_loc_fb[3, 10] = 1
        W_loc_fb[5, 10] = 1

        def add_block_fb(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_row = a_row_start + r
                    global_a_col = a_col_start + c
                    global_a_idx = global_a_row * m + global_a_col
                    U[global_a_idx, col_offset:col_offset + 11] += U_loc_fb[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_row = b_row_start + r
                    global_b_col = b_col_start + c
                    global_b_idx = global_b_row * p + global_b_col
                    V[global_b_idx, col_offset:col_offset + 11] += V_loc_fb[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_row = c_row_start + r
                    global_c_col = c_col_start + c
                    global_c_idx = global_c_col * n + global_c_row
                    W[global_c_idx, col_offset:col_offset + 11] += W_loc_fb[local_c_idx, :]

        add_block_fb(0, 0, 0, 0, 0, 0, 0)
        add_block_fb(1, 0, 2, 2, 0, 0, 0)
        add_block_fb(2, 0, 0, 0, 2, 0, 2)
        add_block_fb(3, 0, 2, 2, 2, 0, 2)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
