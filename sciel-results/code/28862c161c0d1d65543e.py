import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute unfoldings
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def verify_decomp(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_solve(T1, T2, T3, U_init, V_init, W_init, max_iters=500, tol=1e-12):
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        for it in range(max_iters):
            if time.time() > deadline:
                break

            KR_VW = khatri_rao(V, W)
            try:
                U_new, _, _, _ = np.linalg.lstsq(KR_VW, T1.T, rcond=None)
                U_new = U_new.T
            except:
                break

            KR_WU = khatri_rao(W, U)
            try:
                V_new, _, _, _ = np.linalg.lstsq(KR_WU, T2.T, rcond=None)
                V_new = V_new.T
            except:
                break

            KR_UV = khatri_rao(U, V)
            try:
                W_new, _, _, _ = np.linalg.lstsq(KR_UV, T3.T, rcond=None)
                W_new = W_new.T
            except:
                break

            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new

            if change < tol:
                break

        return U, V, W

    def try_round_and_verify(U, V, W):
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomp(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None

    def residual_norm(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - recon)

    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        # Fallback to schoolbook if rank 50 is invalid
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    rng = np.random.default_rng(42)

    # Try to reduce rank to 48
    target_rank = 48
    num_removals = 50 - target_rank
    
    # Try removing 2 columns at a time
    # Heuristic: remove columns with smallest norms
    norms = np.linalg.norm(U50, axis=0) * np.linalg.norm(V50, axis=0) * np.linalg.norm(W50, axis=0)
    idx_to_remove = np.argsort(norms)[:num_removals]
    idx_to_keep = np.setdiff1d(np.arange(50), idx_to_remove)
    
    U_init = U50[:, idx_to_keep].copy()
    V_init = V50[:, idx_to_keep].copy()
    W_init = W50[:, idx_to_keep].copy()

    # Add small noise
    U_init += rng.normal(0, 0.01, U_init.shape)
    V_init += rng.normal(0, 0.01, V_init.shape)
    W_init += rng.normal(0, 0.01, W_init.shape)

    U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_init, V_init, W_init, max_iters=1000)
    err = residual_norm(U_sol, V_sol, W_sol)
    
    if err < 1e-8:
        rounded = try_round_and_verify(U_sol, V_sol, W_sol)
        if rounded is not None:
            best_U, best_V, best_W, best_rank = rounded[0], rounded[1], rounded[2], target_rank
        else:
            best_U, best_V, best_W, best_rank = U_sol, V_sol, W_sol, target_rank

    # If that failed, try random restarts
    if best_rank == 50:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            idx = list(np.random.choice(50, target_rank, replace=False))
            U_init = U50[:, idx].copy()
            V_init = V50[:, idx].copy()
            W_init = W50[:, idx].copy()
            
            U_init += rng.normal(0, 0.05, U_init.shape)
            V_init += rng.normal(0, 0.05, V_init.shape)
            W_init += rng.normal(0, 0.05, W_init.shape)

            U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_init, V_init, W_init, max_iters=500)
            err = residual_norm(U_sol, V_sol, W_sol)
            if err < 1e-8:
                rounded = try_round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    best_U, best_V, best_W, best_rank = rounded[0], rounded[1], rounded[2], target_rank
                    break
                else:
                    best_U, best_V, best_W, best_rank = U_sol, V_sol, W_sol, target_rank
                    break

    if not verify_decomp(best_U, best_V, best_W):
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
