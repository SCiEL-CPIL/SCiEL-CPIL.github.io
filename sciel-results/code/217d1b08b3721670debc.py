import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin from 150s limit
    
    # Target tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if (i, j, k) is valid, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Pre-compute flattened modes for ALS
    T_flat_U = T.reshape(n*m, m*p * p*n)
    T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    def compute_residual(U, V, W):
        """Compute the residual norm for a given decomposition."""
        M = np.einsum('ai,ci->aci', U, W)  # (n*m, p*n, rank)
        T_approx = np.einsum('aci,bi->abc', M, V)  # (n*m, m*p, p*n)
        R = T - T_approx
        return np.sum(R**2)
    
    def svd_init(rank):
        """Initialize U, V, W using SVD of the tensor modes."""
        # SVD of mode 0 unfolding
        U_svd, _, _ = np.linalg.svd(T_flat_U, full_matrices=False)
        U = U_svd[:, :rank]
        
        # SVD of mode 1 unfolding
        V_svd, _, _ = np.linalg.svd(T_flat_V, full_matrices=False)
        V = V_svd[:, :rank]
        
        # SVD of mode 2 unfolding
        W_svd, _, _ = np.linalg.svd(T_flat_W, full_matrices=False)
        W = W_svd[:, :rank]
        
        return U, V, W

    def als_optimize(U, V, W, max_iters=100, tol=1e-12, use_l1=False):
        """Alternating Least Squares optimization with optional L1 sparsity."""
        rank = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            K_U = np.einsum('bi,ci->bci', V, W).reshape(-1, rank)
            A_U = K_U.T @ K_U
            A_U += 1e-10 * np.eye(rank)
            B_U = T_flat_U @ K_U
            try:
                U = np.linalg.solve(A_U.T, B_U.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update V
            K_V = np.einsum('ai,ci->aci', U, W).reshape(-1, rank)
            A_V = K_V.T @ K_V
            A_V += 1e-10 * np.eye(rank)
            B_V = T_flat_V @ K_V
            try:
                V = np.linalg.solve(A_V.T, B_V.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update W
            K_W = np.einsum('ai,bi->abi', U, V).reshape(-1, rank)
            A_W = K_W.T @ K_W
            A_W += 1e-10 * np.eye(rank)
            B_W = T_flat_W @ K_W
            try:
                W = np.linalg.solve(A_W.T, B_W.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Optional L1 sparsity (soft thresholding)
            if use_l1:
                threshold = 0.1 * (1 - it / max_iters)
                U[np.abs(U) < threshold] = 0
                V[np.abs(V) < threshold] = 0
                W[np.abs(W) < threshold] = 0

            if it % 5 == 0:
                residual = compute_residual(U, V, W)
                if residual < tol:
                    break
        
        return U, V, W, compute_residual(U, V, W)
    
    def try_round_and_verify(U, V, W):
        """Try to round to half-integers and verify."""
        # Try rounding to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        M_r = np.einsum('ai,ci->aci', U_rounded, W_rounded)
        T_approx_r = np.einsum('aci,bi->abc', M_r, V_rounded)
        if np.allclose(T_approx_r, T, atol=1e-9):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        
        # Try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        M_i = np.einsum('ai,ci->aci', U_int, W_int)
        T_approx_i = np.einsum('aci,bi->abc', M_i, V_int)
        if np.allclose(T_approx_i, T, atol=1e-9):
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
            
        return None

    # Schoolbook solution (rank 50)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    
    # Try ranks from 49 down to 30
    # We know rank >= max(10, 25, 25) = 25.
    # Schoolbook is 50.
    for rank in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline - 20:
            break
        
        # SVD Initialization is usually best for starting ALS
        U_init, V_init, W_init = svd_init(rank)
        
        # Try a few variations of ALS
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            # Restart 0: Pure ALS from SVD
            U_opt, V_opt, W_opt, residual = als_optimize(U_init, V_init, W_init, max_iters=100, use_l1=False)
            
            # Restart 1: ALS with L1 sparsity
            if time.time() < deadline - 15:
                U_opt2, V_opt2, W_opt2, residual2 = als_optimize(U_init, V_init, W_init, max_iters=100, use_l1=True)
                if residual2 < residual:
                    U_opt, V_opt, W_opt, residual = U_opt2, V_opt2, W_opt2, residual2
            
            # Restart 2: Random perturbation of SVD
            if time.time() < deadline - 15:
                np.random.seed(rank + restart)
                U_pert = U_init + np.random.randn(n*m, rank) * 0.1
                V_pert = V_init + np.random.randn(m*p, rank) * 0.1
                W_pert = W_init + np.random.randn(p*n, rank) * 0.1
                U_opt3, V_opt3, W_opt3, residual3 = als_optimize(U_pert, V_pert, W_pert, max_iters=50, use_l1=False)
                if residual3 < residual:
                    U_opt, V_opt, W_opt, residual = U_opt3, V_opt3, W_opt3, residual3
            
            if residual < 1e-8:
                result = try_round_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    return result
    
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
