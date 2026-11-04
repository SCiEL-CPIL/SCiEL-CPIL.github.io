import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
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

    def get_rank50():
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

    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Compute matricization ranks for lower bound
    T1 = T.reshape(n * m, -1)
    T2 = T.reshape(m * p, -1)
    T3 = T.reshape(p * n, -1)
    rank_lb1 = np.linalg.matrix_rank(T1)
    rank_lb2 = np.linalg.matrix_rank(T2)
    rank_lb3 = np.linalg.matrix_rank(T3)
    rank_lower = max(rank_lb1, rank_lb2, rank_lb3)

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_solve(T1, T2, T3, U, V, W, max_iters=100, tol=1e-12):
        R = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
        return U, V, W

    def polish(U, V, W, max_iters=50):
        """High-precision polish pass."""
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < 1e-14:
                break
        return U, V, W

    def try_eliminate_column(U, V, W, col_idx):
        R = U.shape[1]
        if R <= rank_lower:
            return None
        U_sub = np.delete(U, col_idx, axis=1)
        V_sub = np.delete(V, col_idx, axis=1)
        W_sub = np.delete(W, col_idx, axis=1)
        
        U_new, V_new, W_new = als_solve(T1, T2, T3, U_sub, V_sub, W_sub, max_iters=30, tol=1e-10)
        recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        if np.allclose(recon, T, atol=1e-7):
            return U_new, V_new, W_new
        return None

    def greedy_column_elimination(U, V, W, max_rounds=8):
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            R = U.shape[1]
            if R <= rank_lower:
                break
            eliminated = False
            for col_idx in range(R - 1, -1, -1):
                result = try_eliminate_column(U, V, W, col_idx)
                if result is not None:
                    U, V, W = result
                    eliminated = True
                    break
            if not eliminated:
                break
        return U, V, W

    rng = np.random.default_rng(42)

    # Search from rank 49 down to max(rank_lower, 30)
    min_search = max(rank_lower, 30)
    for target_rank in range(49, min_search - 1, -1):
        if time.time() > deadline - 15:
            break
        found = False
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            if restart == 0:
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.03, U_init.shape)
                V_init += rng.normal(0, 0.03, V_init.shape)
                W_init += rng.normal(0, 0.03, W_init.shape)
            elif restart == 1:
                U_init = U50[:, :target_rank].copy() * rng.uniform(0.5, 1.5, (1, target_rank))
                V_init = V50[:, :target_rank].copy() * rng.uniform(0.5, 1.5, (1, target_rank))
                W_init = W50[:, :target_rank].copy() * rng.uniform(0.5, 1.5, (1, target_rank))
            elif restart == 2:
                Q = np.linalg.qr(rng.normal(0, 1, (target_rank, target_rank)))[0]
                U_init = U50[:, :target_rank] @ Q
                V_init = V50[:, :target_rank] @ Q
                W_init = W50[:, :target_rank] @ Q
            elif restart == 3:
                # Use a subset of columns with random scaling
                idx = rng.choice(50, size=target_rank, replace=False)
                U_init = U50[:, idx].copy()
                V_init = V50[:, idx].copy()
                W_init = W50[:, idx].copy()
                U_init *= rng.uniform(0.8, 1.2, (1, target_rank))
                V_init *= rng.uniform(0.8, 1.2, (1, target_rank))
                W_init *= rng.uniform(0.8, 1.2, (1, target_rank))
            else:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))

            U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_init, V_init, W_init, max_iters=80)

            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-6:
                # Polish
                U_sol, V_sol, W_sol = polish(U_sol, V_sol, W_sol, max_iters=40)
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)

            if verify_decomp(U_sol, V_sol, W_sol):
                U_sol, V_sol, W_sol = greedy_column_elimination(U_sol, V_sol, W_sol)
                final_rank = U_sol.shape[1]
                if verify_decomp(U_sol, V_sol, W_sol):
                    if final_rank < best_rank:
                        best_rank = final_rank
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        found = True
                        break
                elif target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    found = True
                    break

        if found and best_rank <= min_search + 2:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
