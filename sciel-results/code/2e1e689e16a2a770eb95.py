import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    rank_sb = n * m * p  # 63

    # Build the target multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition as guaranteed fallback
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
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
    best_rank = rank_sb

    def get_residual(U, V, W):
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.linalg.norm(T_approx - T)

    def verify(U, V, W, tol=1e-10):
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    def als_update(U, V, W, T, target_rank, max_iters=60, tol=1e-12):
        """ALS with vectorized updates and convergence tracking."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        prev_res = get_residual(U, V, W)

        for it in range(max_iters):
            if time.time() > deadline - 15.0:
                break

            # Update U: for each row i, solve U[i,:] @ (V[:,r]*W[:,r]) = T[i,:,:].flatten()
            # Vectorized: build matrix M_U where M_U[:, r] = kron(V[:,r], W[:,r])
            Vkr = np.zeros((m * p * p * n, target_rank))
            for r in range(target_rank):
                Vkr[:, r] = np.kron(V[:, r], W[:, r])
            VkrT_Vkr = Vkr.T @ Vkr
            VkrT_Vkr += 1e-10 * np.eye(target_rank)  # regularization
            T_flat = T.reshape(n * m, -1)
            U_new = (T_flat @ Vkr) @ np.linalg.pinv(VkrT_Vkr)

            # Update V: for each row j
            UW = np.zeros((n * m * p * n, target_rank))
            for r in range(target_rank):
                UW[:, r] = np.kron(U_new[:, r], W[:, r])
            UW_T_UW = UW.T @ UW
            UW_T_UW += 1e-10 * np.eye(target_rank)
            V_new = np.zeros((m * p, target_rank))
            for j in range(m * p):
                T_j = T[:, j, :].reshape(-1)
                V_new[j] = (T_j @ UW) @ np.linalg.pinv(UW_T_UW)

            # Update W: for each row k
            UV = np.zeros((n * m * m * p, target_rank))
            for r in range(target_rank):
                UV[:, r] = np.kron(U_new[:, r], V_new[:, r])
            UV_T_UV = UV.T @ UV
            UV_T_UV += 1e-10 * np.eye(target_rank)
            W_new = np.zeros((p * n, target_rank))
            for k in range(p * n):
                T_k = T[:, :, k].reshape(-1)
                W_new[k] = (T_k @ UV) @ np.linalg.pinv(UV_T_UV)

            U, V, W = U_new, V_new, W_new
            res = get_residual(U, V, W)

            if abs(prev_res - res) < tol * max(1.0, prev_res):
                break
            prev_res = res

        return U, V, W

    def polish(U, V, W, target_rank, max_iters=100):
        """High-precision polish: more iterations, tighter convergence."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        prev_res = get_residual(U, V, W)

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break

            Vkr = np.zeros((m * p * p * n, target_rank))
            for r in range(target_rank):
                Vkr[:, r] = np.kron(V[:, r], W[:, r])
            VkrT_Vkr = Vkr.T @ Vkr
            VkrT_Vkr += 1e-12 * np.eye(target_rank)
            T_flat = T.reshape(n * m, -1)
            U_new = (T_flat @ Vkr) @ np.linalg.pinv(VkrT_Vkr)

            UW = np.zeros((n * m * p * n, target_rank))
            for r in range(target_rank):
                UW[:, r] = np.kron(U_new[:, r], W[:, r])
            UW_T_UW = UW.T @ UW
            UW_T_UW += 1e-12 * np.eye(target_rank)
            V_new = np.zeros((m * p, target_rank))
            for j in range(m * p):
                T_j = T[:, j, :].reshape(-1)
                V_new[j] = (T_j @ UW) @ np.linalg.pinv(UW_T_UW)

            UV = np.zeros((n * m * m * p, target_rank))
            for r in range(target_rank):
                UV[:, r] = np.kron(U_new[:, r], V_new[:, r])
            UV_T_UV = UV.T @ UV
            UV_T_UV += 1e-12 * np.eye(target_rank)
            W_new = np.zeros((p * n, target_rank))
            for k in range(p * n):
                T_k = T[:, :, k].reshape(-1)
                W_new[k] = (T_k @ UV) @ np.linalg.pinv(UV_T_UV)

            U, V, W = U_new, V_new, W_new
            res = get_residual(U, V, W)
            if abs(prev_res - res) < 1e-14 * max(1.0, prev_res):
                break
            prev_res = res

        return U, V, W

    # Search ranks from just below schoolbook downward
    # Use structured initialization: perturb schoolbook columns
    for target_rank in range(rank_sb - 1, 44, -1):
        if time.time() > deadline - 30.0:
            break

        # Determine number of attempts based on remaining time
        remaining = deadline - 30.0 - time.time()
        if remaining < 5.0:
            break

        num_attempts = min(6, max(2, int(remaining / 5.0)))

        found = False
        for attempt in range(num_attempts):
            if time.time() > deadline - 25.0:
                break

            np.random.seed(42 + target_rank * 137 + attempt * 31)

            # Structured initialization: take first target_rank columns of schoolbook
            # and add small perturbation to escape the schoolbook manifold
            if target_rank <= rank_sb:
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                # Add perturbation to break the sparse structure
                scale = 0.3
                U += np.random.randn(n * m, target_rank) * scale
                V += np.random.randn(m * p, target_rank) * scale
                W += np.random.randn(p * n, target_rank) * scale
            else:
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5

            # ALS
            U, V, W = als_update(U, V, W, T, target_rank, max_iters=50)

            res = get_residual(U, V, W)
            if res < 1e-10:
                # Polish for higher precision
                U, V, W = polish(U, V, W, target_rank, max_iters=80)
                res = get_residual(U, V, W)
                if res < 1e-9:
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = target_rank
                    found = True
                    break
            elif res < 1e-6:
                # Close but not exact - try to polish harder
                U, V, W = polish(U, V, W, target_rank, max_iters=150)
                res = get_residual(U, V, W)
                if res < 1e-9:
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = target_rank
                    found = True
                    break

        if found:
            # Found a solution at this rank; continue to try lower ranks
            # but with fewer attempts since we have a valid solution
            pass

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-8):
        # Fall back to schoolbook
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = rank_sb

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
