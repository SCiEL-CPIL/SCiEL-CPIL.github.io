import numpy as np
import time

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = N, M, P
    
    # Dimensions of the factor matrices
    dim_U = n * m  # 15
    dim_V = m * p  # 35
    dim_W = p * n  # 21
    
    # Target Tensor T*
    # T* is defined by T*[a, b, d] = 1 if a=im+j, b=jp+k, d=kn+i for some i,j,k
    # Else 0.
    # We don't need to build the full (15, 35, 21) tensor explicitly for ALS if we are clever,
    # but it's small enough (15*35*21 = 11025 entries) to build.
    T_star = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_star[a, b, d] = 1.0
    
    # Pre-flatten target slices for ALS
    # For U update: fix V, W. Solve for U.
    # T_star[a, :, :] = sum_c U[a, c] * (V[:, c] outer W[:, c])
    # Let Z_c = V[:, c] * W[:, c] (flattened)
    # Then T_star[a, :, :].flat = sum_c U[a, c] * Z_c
    # So [Z_1 ... Z_R] @ U[a, :] = T_star[a, :, :].flat
    
    # For V update: fix U, W.
    # T_star[:, b, :] = sum_c V[b, c] * (U[:, c] outer W[:, c])
    # Let Y_c = U[:, c] * W[:, c]
    # [Y_1 ... Y_R] @ V[b, :] = T_star[:, b, :].flat
    
    # For W update: fix U, V.
    # T_star[:, :, d] = sum_c W[d, c] * (U[:, c] outer V[:, c])
    # Let X_c = U[:, c] * V[:, c]
    # [X_1 ... X_R] @ W[d, :] = T_star[:, :, d].flat

    def run_als(rank, max_iter=100, tol=1e-10, seed=42):
        np.random.seed(seed)
        # Random initialization
        U = np.random.randn(dim_U, rank)
        V = np.random.randn(dim_V, rank)
        W = np.random.randn(dim_W, rank)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # Z_c = V[:, c] * W[:, c]
            # Z shape: (rank, dim_V * dim_W)
            Z = np.einsum('jc,kc->jkc', V, W).reshape(rank, -1).T # (dim_V*dim_W, rank)
            # Solve [Z] @ U[a, :] = T_star[a, :, :].flat for all a
            # T_star_flat_U shape: (dim_U, dim_V*dim_W)
            T_flat_U = T_star.reshape(dim_U, -1)
            # Least squares: U = pinv(Z) @ T_flat_U.T ?
            # Z is (dim_V*dim_W, rank). T_flat_U is (dim_U, dim_V*dim_W).
            # We want U (dim_U, rank) such that U @ Z.T = T_flat_U
            # U = T_flat_U @ pinv(Z.T)
            # Or solve Z.T @ U.T = T_flat_U.T
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z.T, T_flat_U.T, rcond=None)
                U = U_new.T
            except np.linalg.LinAlgError:
                pass
            
            # Update V
            # Y_c = U[:, c] * W[:, c]
            Y = np.einsum('ic,kc->ikc', U, W).reshape(rank, -1).T # (dim_U*dim_W, rank)
            T_flat_V = np.moveaxis(T_star, 1, 0).reshape(dim_V, -1) # (dim_V, dim_U*dim_W)
            try:
                V_new, _, _, _ = np.linalg.lstsq(Y.T, T_flat_V.T, rcond=None)
                V = V_new.T
            except np.linalg.LinAlgError:
                pass
            
            # Update W
            # X_c = U[:, c] * V[:, c]
            X = np.einsum('ic,jc->ijc', U, V).reshape(rank, -1).T # (dim_U*dim_V, rank)
            T_flat_W = np.moveaxis(T_star, 2, 0).reshape(dim_W, -1) # (dim_W, dim_U*dim_V)
            try:
                W_new, _, _, _ = np.linalg.lstsq(X.T, T_flat_W.T, rcond=None)
                W = W_new.T
            except np.linalg.LinAlgError:
                pass
            
            # Calculate Error
            # T_approx = sum_c U[:,c] outer V[:,c] outer W[:,c]
            T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
            error = np.linalg.norm(T_approx - T_star)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < tol:
                break
                
        return best_U, best_V, best_W, best_error

    def check_and_round(U, V, W, tol=1e-6):
        """Check if decomposition is exact and try to round to half-integers."""
        # Verify exactness with current floats
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        if not np.allclose(T_approx, T_star, atol=tol):
            return None, None, None, False
            
        # Try rounding to half-integers
        # U, V, W entries should be integer or half-integer
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        T_approx_round = np.einsum('ac,bc,dc->abd', U_round, V_round, W_round)
        if np.allclose(T_approx_round, T_star, atol=1e-9):
            return U_round, V_round, W_round, True
            
        # If rounding fails, return floats if error is tiny, else fail
        if np.max(np.abs(T_approx - T_star)) < 1e-10:
            return U, V, W, True
        return None, None, None, False

    best_result = None
    best_rank = float('inf')
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((dim_U, rank_sb))
    V_sb = np.zeros((dim_V, rank_sb))
    W_sb = np.zeros((dim_W, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try lower ranks
    # Lower bound is max(nm, mp, pn) = 35
    ranks_to_try = [35, 36, 38, 40, 42, 45, 50, 55, 60]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Try multiple seeds for robustness
        for seed in [42, 123, 999]:
            if time.time() > deadline - 10:
                break
            U, V, W, err = run_als(rank, max_iter=50, seed=seed)
            if err < 1e-8:
                U_r, V_r, W_r, ok = check_and_round(U, V, W)
                if ok:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r, rank)
                        # If we found a very low rank, we can stop or continue to see if even lower is possible
                        # But usually minimal rank is hard to beat by much.
                        # Let's break out of seed loop if we found a good one
                        break
                        
    # Convert best result to list of lists for output
    if best_result is not None:
        U, V, W, r = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
