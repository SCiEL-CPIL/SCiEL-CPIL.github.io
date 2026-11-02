import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135  # safety margin before 150s limit

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1.0

    # Strassen vectors for 2x2x2
    u_vecs = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    v_vecs = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    w_vecs = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)

    # Build rank-56 Strassen solution: 8 blocks of 7
    rank_56 = 56
    U56 = np.zeros((n * m, rank_56))
    V56 = np.zeros((m * p, rank_56))
    W56 = np.zeros((p * n, rank_56))

    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            # U: 4 entries for A (2x2)
            U56[:, col] = u_vecs[s]
            # V: 32 entries for B (2x16), only 4 non-zero for block r
            v_indices = [2 * r, 2 * r + 1, 16 + 2 * r, 16 + 2 * r + 1]
            V56[v_indices, col] = v_vecs[s]
            # W: 32 entries for C (16x2), only 4 non-zero for block r
            w_indices = [4 * r, 4 * r + 2, 4 * r + 1, 4 * r + 3]
            W56[w_indices, col] = w_vecs[s]

    def _verify(U, V, W, atol=1e-6):
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.allclose(T_recon, T, atol=atol)

    # Verify rank-56 solution
    valid_56 = _verify(U56, V56, W56)
    
    best_U, best_V, best_W = U56, V56, W56
    best_rank = rank_56

    if not valid_56:
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}

    def _als_solve(r, U_init, V_init, W_init, max_iter=40, tol=1e-9):
        """ALS optimization for rank-r decomposition."""
        Ua = np.array(U_init, dtype=float).copy()
        Va = np.array(V_init, dtype=float).copy()
        Wa = np.array(W_init, dtype=float).copy()

        for it in range(max_iter):
            if time.time() > deadline - 15:
                return None, None, None

            # Update U
            KR_VW = np.einsum('bs,cs->bcs', Va, Wa).reshape(m * p * p * n, r)
            for a in range(n * m):
                T_a = T[a].reshape(-1)
                Ua[a, :], _, _, _ = np.linalg.lstsq(KR_VW, T_a, rcond=None)

            # Update V
            KR_UW = np.einsum('as,cs->acs', Ua, Wa).reshape(n * m * p * n, r)
            for b in range(m * p):
                T_b = T[:, b, :].reshape(-1)
                Va[b, :], _, _, _ = np.linalg.lstsq(KR_UW, T_b, rcond=None)

            # Update W
            KR_UV = np.einsum('as,bs->abs', Ua, Va).reshape(n * m * m * p, r)
            for c in range(p * n):
                T_c = T[:, :, c].reshape(-1)
                Wa[c, :], _, _, _ = np.linalg.lstsq(KR_UV, T_c, rcond=None)

            # Check convergence
            T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                break

        # Final check
        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        if np.allclose(T_recon, T, atol=1e-5):
            return Ua, Va, Wa
        return None, None, None

    # Try to find lower rank solutions
    for target_rank in range(55, 40, -1):
        if time.time() > deadline - 25:
            break

        success = False
        for init_seed in range(5):
            if time.time() > deadline - 15:
                break

            np.random.seed(init_seed * 7919 + 42)

            if target_rank == 55:
                # Remove one column and re-optimize
                U_init = U56[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.01
                V_init = V56[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.01
                W_init = W56[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.01
            elif target_rank >= 49:
                # Remove a few columns
                U_init = U56[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.05
                V_init = V56[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.05
                W_init = W56[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.05
            else:
                # Random initialization
                U_init = np.random.randn(n * m, target_rank) * 0.1
                V_init = np.random.randn(m * p, target_rank) * 0.1
                W_init = np.random.randn(p * n, target_rank) * 0.1

            U_res, V_res, W_res = _als_solve(target_rank, U_init, V_init, W_res if W_res is not None else W_init, max_iter=30)

            if U_res is not None:
                best_U, best_V, best_W = U_res, V_res, W_res
                best_rank = target_rank
                success = True
                break

        if success:
            # Try to go even lower
            continue
        else:
            # If we couldn't achieve this rank, break
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
