import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

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
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()

    def als_reduce(U_init, V_init, W_init, target_rank, max_iters=50, tol=1e-10):
        """Fast ALS to reduce rank. Returns (U, V, W, success)."""
        R = target_rank
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        T_2d_a = T.reshape(n*m, -1)  # 8 x 392
        T_2d_b = T.reshape(m*p, -1)  # 28 x 112
        T_2d_c = T.reshape(p*n, -1)  # 14 x 224
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                return U, V, W, False
            
            # Solve for U: T[a, b*c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # => T_2d_a = U @ (V ⊙ W)^T where (V⊙W)[:,r] = kron(V[:,r], W[:,r])
            KR_VW = np.zeros((m*p*p*n, R))
            for r in range(R):
                KR_VW[:, r] = np.kron(V[:, r], W[:, r])
            U_new = T_2d_a @ np.linalg.pinv(KR_VW)
            
            KR_WU = np.zeros((p*n*n*m, R))
            for r in range(R):
                KR_WU[:, r] = np.kron(W[:, r], U_new[:, r])
            V_new = T_2d_b @ np.linalg.pinv(KR_WU)
            
            KR_UV = np.zeros((n*m*m*p, R))
            for r in range(R):
                KR_UV[:, r] = np.kron(U_new[:, r], V_new[:, r])
            W_new = T_2d_c @ np.linalg.pinv(KR_UV)
            
            change = (np.linalg.norm(U_new - U) + 
                     np.linalg.norm(V_new - V) + 
                     np.linalg.norm(W_new - W))
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        
        if verify_decomp(U, V, W, atol=1e-8):
            return U, V, W, True
        return U, V, W, False

    def try_rank(target_rank, num_restarts=5):
        """Try to find a valid decomposition of given rank."""
        rng = np.random.default_rng(target_rank * 1000 + 42)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize from rank-50, truncate + perturb
            scale = 0.5 / (restart + 1)
            U_init = U50[:, :target_rank].copy() + rng.normal(0, scale, (n*m, target_rank))
            V_init = V50[:, :target_rank].copy() + rng.normal(0, scale, (m*p, target_rank))
            W_init = W50[:, :target_rank].copy() + rng.normal(0, scale, (p*n, target_rank))
            
            U_sol, V_sol, W_sol, success = als_reduce(U_init, V_init, W_init, target_rank, max_iters=80)
            if success:
                # Try to round to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                if verify_decomp(U_r, V_r, W_r, atol=1e-8):
                    return U_r, V_r, W_r
                return U_sol, V_sol, W_sol
        
        return None

    # Try to reduce rank
    for target_rank in range(49, 25, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, num_restarts=3)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol

    # Final verification
    if not verify_decomp(best_U, best_V, best_W, atol=1e-9):
        if verify_decomp(U50, V50, W50, atol=1e-9):
            best_U, best_V, best_W = U50, V50, W50
            best_rank = 50

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
