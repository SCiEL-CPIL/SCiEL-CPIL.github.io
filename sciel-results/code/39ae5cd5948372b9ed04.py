"""Optimized matrix multiplication decomposition for 4x4 * 4x9."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.max(np.abs(T - 1.0)) < tol

    def try_find_decomp(n_dim, m_dim, p_dim, target_rank, max_steps=3000, n_trials=3):
        """Try to find a rank-target decomposition for n_dim x m_dim * m_dim x p_dim."""
        nm_d = n_dim * m_dim
        mp_d = m_dim * p_dim
        pn_d = p_dim * n_dim

        for trial in range(n_trials):
            if time.time() > deadline:
                return None

            # Initialize with small random values
            scale = 0.1
            U_d = np.random.randn(nm_d, target_rank) * scale
            V_d = np.random.randn(mp_d, target_rank) * scale
            W_d = np.random.randn(pn_d, target_rank) * scale

            # Adam optimizer
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_U = np.zeros_like(U_d)
            v_U = np.zeros_like(U_d)
            m_V = np.zeros_like(V_d)
            v_V = np.zeros_like(V_d)
            m_W = np.zeros_like(W_d)
            v_W = np.zeros_like(W_d)
            t = 0
            lr = 0.02

            for step in range(max_steps):
                if time.time() > deadline:
                    return None

                t += 1
                U_3d = U_d.reshape(n_dim, m_dim, target_rank)
                V_3d = V_d.reshape(m_dim, p_dim, target_rank)
                W_3d = W_d.reshape(p_dim, n_dim, target_rank)

                # Compute tensor
                UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W_3d)
                err = T - 1.0

                # Gradients
                grad_U = np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                grad_V = np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                grad_W = np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)

                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U**2
                U_d -= lr * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)

                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V**2
                V_d -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W**2
                W_d -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)

                if step % 500 == 0:
                    max_err = np.max(np.abs(T))
                    if max_err < 1e-8:
                        return U_d, V_d, W_d

            # Final check
            U_3d = U_d.reshape(n_dim, m_dim, target_rank)
            V_3d = V_d.reshape(m_dim, p_dim, target_rank)
            W_3d = W_d.reshape(p_dim, n_dim, target_rank)
            UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
            T = np.einsum('ijkr,kir->ijk', UV, W_3d)
            if np.max(np.abs(T - 1.0)) < 1e-8:
                return U_d, V_d, W_d

        return None

    # Strategy 1: Try to find a better decomposition for 4x4*4x4, combine with 4x4*4x5 schoolbook
    # 4x4*4x4 schoolbook rank = 64, 4x4*4x5 schoolbook rank = 80, total = 144
    # If we can find 4x4*4x4 with rank < 64, we save
    
    # Known: Strassen gives 49 for 4x4*4x4
    # Let's try ranks from 49 up to 63

    for target_rank in range(49, 64):
        if time.time() > deadline:
            break

        result = try_find_decomp(4, 4, 4, target_rank, max_steps=2000, n_trials=2)
        if result is not None:
            U4, V4, W4 = result

            # Combine with 4x4*4x5 schoolbook (rank 80)
            total_rank = target_rank + 80

            if total_rank < best_rank:
                U_new = np.zeros((nm, total_rank))
                V_new = np.zeros((mp, total_rank))
                W_new = np.zeros((pn, total_rank))

                # First 4x4*4x4 part
                U_new[:, :target_rank] = U4
                V_new[:16, :target_rank] = V4
                W_new[:16, :target_rank] = W4

                # 4x4*4x5 schoolbook part
                col = target_rank
                for i in range(4):
                    for j in range(4):
                        for k in range(4, 9):
                            U_new[i * 4 + j, col] = 1
                            V_new[j * 9 + k, col] = 1
                            W_new[k * 4 + i, col] = 1
                            col += 1

                if verify_decomposition(U_new, V_new, W_new):
                    best_rank = total_rank
                    best_U = U_new.copy()
                    best_V = V_new.copy()
                    best_W = W_new.copy()
                break

    # Strategy 2: Try direct optimization on the full 4x4*4x9 problem
    # Try ranks below current best
    if time.time() < deadline:
        for target_rank in range(best_rank - 1, 120, -1):
            if time.time() > deadline:
                break

            result = try_find_decomp(4, 4, 9, target_rank, max_steps=1500, n_trials=1)
            if result is not None:
                U_d, V_d, W_d = result
                if verify_decomposition(U_d, V_d, W_d):
                    best_rank = target_rank
                    best_U = U_d.copy()
                    best_V = V_d.copy()
                    best_W = W_d.copy()
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
