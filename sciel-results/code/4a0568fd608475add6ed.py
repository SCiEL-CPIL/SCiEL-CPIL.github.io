import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for evaluator's 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
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
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
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

    T1 = T.reshape(n * m, -1)
    T2 = T.reshape(m * p, -1)
    T3 = T.reshape(p * n, -1)

    def khatri_rao(A, B):
        """
        Vectorized Khatri-Rao product.
        A: (M, R), B: (N, R)
        Returns: (M*N, R)
        """
        M, R = A.shape
        N, _ = B.shape
        return (A[:, np.newaxis, :] * B[np.newaxis, :, :]).reshape(M * N, R)

    def als_solve(T1, T2, T3, U, V, W, max_iters=100, tol=1e-10, deadline=None):
        R = U.shape[1]
        prev_residual = np.inf
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 3:
                break
            # Update U
            KR_VW = khatri_rao(V, W)
            try:
                U = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            # Update V
            KR_WU = khatri_rao(W, U)
            try:
                V = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            # Update W
            KR_UV = khatri_rao(U, V)
            try:
                W = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
            if abs(prev_residual - residual) < 1e-12 * max(1.0, prev_residual):
                break
            prev_residual = residual
        return U, V, W

    def get_svd_init(target_rank, T1, T2, T3):
        """Get SVD-based initialization for ALS."""
        # SVD of matricizations
        U_svd, _, _ = np.linalg.svd(T1, full_matrices=False)
        V_svd, _, _ = np.linalg.svd(T2, full_matrices=False)
        W_svd, _, _ = np.linalg.svd(T3, full_matrices=False)
        
        U_init = U_svd[:, :target_rank]
        V_init = V_svd[:, :target_rank]
        W_init = W_svd[:, :target_rank]
        
        return U_init, V_init, W_init

    def try_reduce_rank(U_prev, V_prev, W_prev, target_rank, rng, deadline):
        """
        Attempt to find a solution of target_rank using U_prev, V_prev, W_prev as guide.
        """
        if target_rank >= U_prev.shape[1]:
            return None
            
        U_init = None
        V_init = None
        W_init = None
        
        # Try multiple initialization strategies
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            if restart == 0:
                # SVD-based initialization
                U_init, V_init, W_init = get_svd_init(target_rank, T1, T2, T3)
                
            elif restart == 1:
                # Copy a subset of the previous solution's columns, shuffled
                cols = rng.choice(U_prev.shape[1], target_rank, replace=False)
                U_init = U_prev[:, cols].copy()
                V_init = V_prev[:, cols].copy()
                W_init = W_prev[:, cols].copy()
                
                # Add small noise to escape local minima
                U_init += rng.normal(0, 0.01, U_init.shape)
                V_init += rng.normal(0, 0.01, V_init.shape)
                W_init += rng.normal(0, 0.01, W_init.shape)
                
            elif restart == 2:
                # Random dense
                U_init = rng.normal(0, 1, (n * m, target_rank))
                V_init = rng.normal(0, 1, (m * p, target_rank))
                W_init = rng.normal(0, 1, (p * n, target_rank))
            elif restart == 3:
                # Hybrid: Mix of previous and random
                U_init = U_prev[:, :target_rank].copy() * 0.5 + rng.normal(0, 0.5, (n * m, target_rank))
                V_init = V_prev[:, :target_rank].copy() * 0.5 + rng.normal(0, 0.5, (m * p, target_rank))
                W_init = W_prev[:, :target_rank].copy() * 0.5 + rng.normal(0, 0.5, (p * n, target_rank))
            else:
                # Sparse random initialization
                U_init = rng.choice([0, 1], (n * m, target_rank)) * rng.choice([-1, 1], (n * m, target_rank))
                V_init = rng.choice([0, 1], (m * p, target_rank)) * rng.choice([-1, 1], (m * p, target_rank))
                W_init = rng.choice([0, 1], (p * n, target_rank)) * rng.choice([-1, 1], (p * n, target_rank))

            U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_init, V_init, W_init, max_iters=80, tol=1e-8, deadline=deadline)
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-6:
                # Polish with tighter tolerance
                U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_sol, V_sol, W_sol, max_iters=50, tol=1e-12, deadline=deadline)
                if verify_decomp(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
        return None

    rng = np.random.default_rng(42)

    # Try to reduce rank from 50 down to 30
    current_U, current_V, current_W = U50, V50, W50
    current_rank = 50

    for target_rank in range(49, 30, -1):
        if time.time() > deadline - 10:
            break
            
        result = try_reduce_rank(current_U, current_V, current_W, target_rank, rng, deadline)
        if result is not None:
            U_sol, V_sol, W_sol = result
            if verify_decomp(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                current_U, current_V, current_W = U_sol, V_sol, W_sol
                current_rank = target_rank
            else:
                pass
        else:
            # If we fail to reduce, try a few more restarts for this specific target_rank
            found = False
            for _ in range(2):
                if time.time() > deadline - 5:
                    break
                result = try_reduce_rank(current_U, current_V, current_W, target_rank, rng, deadline)
                if result is not None:
                    U_sol, V_sol, W_sol = result
                    if verify_decomp(U_sol, V_sol, W_sol):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        current_U, current_V, current_W = U_sol, V_sol, W_sol
                        current_rank = target_rank
                        found = True
                        break
            if not found:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
