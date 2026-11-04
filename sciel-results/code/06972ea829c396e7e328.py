import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 144
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook

    def verify_full(U, V, W, rank):
        """Verify that the decomposition is exact for the full n x m x p problem."""
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # Compute tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def verify_sub(U, V, W, n_sub, m_sub, p_sub):
        """Verify sub-tensor decomposition."""
        U_r = U.reshape(n_sub, m_sub, -1)
        V_r = V.reshape(m_sub, p_sub, -1)
        W_r = W.reshape(p_sub, n_sub, -1)
        T = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def is_half_integer(arr):
        """Check if all entries are multiples of 0.5."""
        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-9)

    def try_block_decomposition(n_sub, m_sub, p_sub, r_target, max_restarts=200):
        """
        Try to find a rank-r decomposition for an n_sub x m_sub x p_sub sub-tensor.
        Returns (U, V, W) if found, else None.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None

            # Random binary U and V
            np.random.seed(restart * 1000 + n_sub * 100 + m_sub * 10 + p_sub)
            U_try = np.random.randint(0, 2, (n_sub * m_sub, r_target))
            V_try = np.random.randint(0, 2, (m_sub * p_sub, r_target))

            # Build linear system for W
            num_vars = p_sub * n_sub * r_target
            num_eqs = n_sub * m_sub * p_sub

            A_mat = np.zeros((num_eqs, num_vars))
            b_vec = np.ones(num_eqs)

            eq_idx = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        # sum_a U[i,j,a] * V[j,k,a] * W[k,i,a] = 1
                        for a in range(r_target):
                            coeff = U_try[i * m_sub + j, a] * V_try[j * p_sub + k, a]
                            if coeff != 0:
                                var_idx = (k * n_sub + i) * r_target + a
                                A_mat[eq_idx, var_idx] = coeff
                        eq_idx += 1

            # Solve least squares
            try:
                W_flat, residuals, rank_val, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue

            W_try = W_flat.reshape(p_sub * n_sub, r_target)

            # Check if W is half-integer and verify
            if is_half_integer(W_try):
                if verify_sub(U_try, V_try, W_try, n_sub, m_sub, p_sub):
                    return U_try, V_try, W_try

        return None

    # Strategy 1: Try to decompose into three 4x4x3 blocks with rank 45 each (total 135)
    # This exploits the block structure of the 4x4x9 problem
    n_sub, m_sub, p_sub = 4, 4, 3
    r_sub_target = 45

    if time.time() < deadline - 5:
        result = try_block_decomposition(n_sub, m_sub, p_sub, r_sub_target, max_restarts=100)
        if result is not None:
            U_sub, V_sub, W_sub = result
            best_rank = r_sub_target * 3  # 135

            U_final = np.zeros((n * m, best_rank))
            V_final = np.zeros((m * p, best_rank))
            W_final = np.zeros((p * n, best_rank))

            for block in range(3):
                offset = block * r_sub_target
                U_final[:, offset:offset + r_sub_target] = U_sub
                for j in range(m):
                    for k_local in range(p_sub):
                        k_global = block * p_sub + k_local
                        idx_v = j * p + k_global
                        idx_v_sub = j * p_sub + k_local
                        V_final[idx_v, offset:offset + r_sub_target] = V_sub[idx_v_sub]
                for k_local in range(p_sub):
                    k_global = block * p_sub + k_local
                    for i in range(n):
                        idx_w = k_global * n + i
                        idx_w_sub = k_local * n_sub + i
                        W_final[idx_w, offset:offset + r_sub_target] = W_sub[idx_w_sub]

            # Verify the full decomposition
            if verify_full(U_final, V_final, W_final, best_rank):
                best_U = U_final
                best_V = V_final
                best_W = W_final

    # Strategy 2: If strategy 1 didn't work, try lower rank sub-decompositions
    if best_rank == rank_schoolbook and time.time() < deadline - 5:
        # Try rank 44 for sub-tensor (total 132)
        for r_try in [44, 43, 42]:
            if time.time() > deadline - 5:
                break
            result = try_block_decomposition(n_sub, m_sub, p_sub, r_try, max_restarts=50)
            if result is not None:
                U_sub, V_sub, W_sub = result
                new_rank = r_try * 3

                U_final = np.zeros((n * m, new_rank))
                V_final = np.zeros((m * p, new_rank))
                W_final = np.zeros((p * n, new_rank))

                for block in range(3):
                    offset = block * r_try
                    U_final[:, offset:offset + r_try] = U_sub
                    for j in range(m):
                        for k_local in range(p_sub):
                            k_global = block * p_sub + k_local
                            idx_v = j * p + k_global
                            idx_v_sub = j * p_sub + k_local
                            V_final[idx_v, offset:offset + r_try] = V_sub[idx_v_sub]
                    for k_local in range(p_sub):
                        k_global = block * p_sub + k_local
                        for i in range(n):
                            idx_w = k_global * n + i
                            idx_w_sub = k_local * n_sub + i
                            W_final[idx_w, offset:offset + r_try] = W_sub[idx_w_sub]

                if verify_full(U_final, V_final, W_final, new_rank):
                    best_U = U_final
                    best_V = V_final
                    best_W = W_final
                    best_rank = new_rank
                    break

    # Strategy 3: Try a direct full-problem search with rank 130
    if best_rank == rank_schoolbook and time.time() < deadline - 10:
        r_full_target = 130
        np.random.seed(12345)

        for restart in range(30):
            if time.time() > deadline - 5:
                break

            U_try = np.random.randint(0, 2, (n * m, r_full_target))
            V_try = np.random.randint(0, 2, (m * p, r_full_target))

            num_vars = p * n * r_full_target
            num_eqs = n * m * p

            A_mat = np.zeros((num_eqs, num_vars))
            b_vec = np.ones(num_eqs)

            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for a in range(r_full_target):
                            coeff = U_try[i * m + j, a] * V_try[j * p + k, a]
                            if coeff != 0:
                                var_idx = (k * n + i) * r_full_target + a
                                A_mat[eq_idx, var_idx] = coeff
                        eq_idx += 1

            try:
                W_flat, residuals, rank_val, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue

            W_try = W_flat.reshape(p * n, r_full_target)

            if is_half_integer(W_try):
                if verify_full(U_try, V_try, W_try, r_full_target):
                    best_U = U_try
                    best_V = V_try
                    best_W = W_try
                    best_rank = r_full_target
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
