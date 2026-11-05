import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    # 1. Construct Schoolbook Decomposition (Rank 96)
    # This is guaranteed to work and takes negligible time.
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school

    # 2. Build Target Tensor for verification/search
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check(U, V, W, tol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    # 3. Attempt to find Rank 95 decomposition
    # Use a robust optimization approach: Gauss-Newton or Levenberg-Marquardt
    # We will flatten the problem and use scipy.optimize.least_squares if available, 
    # or a manual Gauss-Newton step.
    
    target_rank = 95
    if time.time() < deadline - 10:
        # Initialize with random Gaussian, scaled
        np.random.seed(42)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        # Run a few iterations of ALS with SVD polishing
        # SVD polishing: After updating one factor, truncate to rank R
        for it in range(50):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed V, W, T[a, :, :] = U[a, :] (V^T W) ? No.
            # T[a] = sum_r U[a, r] (V[:, r] W[:, r]^T)
            # This is a linear system for U[a, :]
            # M = V * W.T (element-wise outer product per r)
            # Actually, let's use the standard ALS normal equations
            # A_U = (V.T @ V) * (W.T @ W) is WRONG.
            # A_U[r1, r2] = sum_{b,c} V[b, r1] V[b, r2] W[c, r1] W[c, r2]
            
            A_U = np.einsum('br,cr,bs,cs->rs', V, W, V, W)
            B_U = np.einsum('abc,br,cr->ar', T, V, W)
            try:
                U_new = np.linalg.solve(A_U + 1e-10 * np.eye(target_rank), B_U.T).T
            except:
                U_new = np.linalg.lstsq(A_U, B_U.T, rcond=None)[0].T
            
            # SVD Polish U
            U_new = np.linalg.svd(U_new, full_matrices=False)[0] @ np.diag(np.linalg.svd(U_new, full_matrices=False)[1]) @ np.linalg.svd(U_new, full_matrices=False)[2].T
            # Actually, just keep the top R components
            Uu, Us, Ut = np.linalg.svd(U_new, full_matrices=False)
            U = Uu[:, :target_rank] * Us[:target_rank]
            
            # Update V
            A_V = np.einsum('ar,cr,as,cs->rs', U, W, U, W)
            B_V = np.einsum('abc,ar,cr->br', T, U, W)
            try:
                V_new = np.linalg.solve(A_V + 1e-10 * np.eye(target_rank), B_V.T).T
            except:
                V_new = np.linalg.lstsq(A_V, B_V.T, rcond=None)[0].T
            Vu, Vs, Vt = np.linalg.svd(V_new, full_matrices=False)
            V = Vu[:, :target_rank] * Vs[:target_rank]
            
            # Update W
            A_W = np.einsum('ar,br,as,bs->rs', U, V, U, V)
            B_W = np.einsum('abc,ar,br->cr', T, U, V)
            try:
                W_new = np.linalg.solve(A_W + 1e-10 * np.eye(target_rank), B_W.T).T
            except:
                W_new = np.linalg.lstsq(A_W, B_W.T, rcond=None)[0].T
            Wu, Ws, Wt = np.linalg.svd(W_new, full_matrices=False)
            W = Wu[:, :target_rank] * Ws[:target_rank]
            
            # Check error
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-8:
                # Found a valid decomposition with rank 95
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break

    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
