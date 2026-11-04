"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Build the schoolbook decomposition (always correct)
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build target tensor as numpy array for efficient computation
    import numpy as np
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def _verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T."""
        # Compute sum over rank of U[:,r] * V[:,r] * W[:,r]
        # Result should equal T
        r = len(U[0])
        recon = np.zeros_like(T)
        for r_idx in range(r):
            u = np.array([U[i][r_idx] for i in range(n * m)], dtype=np.float64)
            v = np.array([V[i][r_idx] for i in range(m * p)], dtype=np.float64)
            w = np.array([W[i][r_idx] for i in range(p * n)], dtype=np.float64)
            recon += np.einsum('a,b,c->abc', u, v, w)
        return np.max(np.abs(recon - T)) < tol
    
    def _try_alternating_projection(rank, max_iter=500, time_limit=30.0):
        """Try to find a rank-rank decomposition via alternating projections."""
        if time.time() > deadline - time_limit:
            return None
        
        # Initialize with random factors
        best_loss = float('inf')
        best_factors = None
        
        for attempt in range(3):
            if time.time() > deadline - time_limit:
                break
            
            # Initialize: take first 'rank' schoolbook columns as starting point
            # and add small random perturbation
            U_init = np.zeros((n * m, rank))
            V_init = np.zeros((m * p, rank))
            W_init = np.zeros((p * n, rank))
            
            for r_idx in range(rank):
                # Use schoolbook columns as initialization
                U_init[:, r_idx] = np.array([U_sb[i][r_idx] for i in range(n * m)], dtype=np.float64)
                V_init[:, r_idx] = np.array([V_sb[i][r_idx] for i in range(m * p)], dtype=np.float64)
                W_init[:, r_idx] = np.array([W_sb[i][r_idx] for i in range(p * n)], dtype=np.float64)
            
            # Add small random perturbation to break symmetry
            if attempt > 0:
                scale = 0.01 * (attempt + 1)
                U_init += scale * np.random.randn(n * m, rank)
                V_init += scale * np.random.randn(m * p, rank)
                W_init += scale * np.random.randn(p * n, rank)
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(max_iter):
                if time.time() > deadline - time_limit:
                    break
                
                # Compute current reconstruction error
                recon = np.zeros_like(T)
                for r_idx in range(rank):
                    recon += np.einsum('a,b,c->abc', U_cur[:, r_idx], V_cur[:, r_idx], W_cur[:, r_idx])
                loss = np.sum((recon - T) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_factors = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                
                if loss < 1e-10:
                    break
                
                # Alternating projection: update each factor
                # For U: fix V and W, solve least squares for U
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # So: T[a,:,:] = U[a,:] @ diag(V*W) where V*W is outer product per r
                
                # Build the "effective" matrix: for each r, eff_r = V[:,r] * W[:,r] (element-wise)
                # Then T[a, b, c] = sum_r U[a,r] * eff_r[b,c]
                # Reshape: T_flat[a, bc] = sum_r U[a,r] * eff_flat[r, bc]
                # So T_flat = U @ eff_flat.T where eff_flat has shape (rank, m*p*p*n)
                
                # This is expensive. Let's use a simpler gradient-based update instead.
                
                # Gradient of loss w.r.t. U[a,r]:
                # d/da,r sum_{b,c} (T[a,b,c] - sum_r' U[a,r']*V[b,r']*W[c,r'])^2
                # = -2 * sum_{b,c} (T[a,b,c] - recon[a,b,c]) * V[b,r] * W[c,r]
                
                # Compute residuals
                residual = T - recon
                
                # Update U
                for a in range(n * m):
                    for r_idx in range(rank):
                        # grad = -2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                        # = -2 * np.einsum('bc,b,c->', residual[a,:,:], V_cur[:,r_idx], W_cur[:,r_idx])
                        grad_val = -2.0 * np.sum(residual[a, :, :] * V_cur[:, r_idx][:, None] * W_cur[:, r_idx][None, :])
                        U_cur[a, r_idx] -= 0.01 * grad_val
                
                # Update V
                recon = np.zeros_like(T)
                for r_idx in range(rank):
                    recon += np.einsum('a,b,c->abc', U_cur[:, r_idx], V_cur[:, r_idx], W_cur[:, r_idx])
                residual = T - recon
                for b in range(m * p):
                    for r_idx in range(rank):
                        grad_val = -2.0 * np.sum(residual[:, b, :] * U_cur[:, r_idx][:, None] * W_cur[:, r_idx][None, :])
                        V_cur[b, r_idx] -= 0.01 * grad_val
                
                # Update W
                recon = np.zeros_like(T)
                for r_idx in range(rank):
                    recon += np.einsum('a,b,c->abc', U_cur[:, r_idx], V_cur[:, r_idx], W_cur[:, r_idx])
                residual = T - recon
                for c in range(p * n):
                    for r_idx in range(rank):
                        grad_val = -2.0 * np.sum(residual[:, :, c] * U_cur[:, r_idx][:, None] * V_cur[:, r_idx][None, :])
                        W_cur[c, r_idx] -= 0.01 * grad_val
                
                # Project to nearest half-integer values
                U_cur = np.round(U_cur * 2) / 2
                V_cur = np.round(V_cur * 2) / 2
                W_cur = np.round(W_cur * 2) / 2
            
            # Check final result
            if best_factors is not None:
                U_check, V_check, W_check = best_factors
                # Convert to lists
                U_list = [[float(U_check[i, r_idx]) for r_idx in range(rank)] for i in range(n * m)]
                V_list = [[float(V_check[i, r_idx]) for r_idx in range(rank)] for i in range(m * p)]
                W_list = [[float(W_check[i, r_idx]) for r_idx in range(rank)] for i in range(p * n)]
                
                if _verify_decomposition(U_list, V_list, W_list, tol=1e-6):
                    return (U_list, V_list, W_list)
        
        return None
    
    # Try to find a decomposition with rank < 72
    for target_rank in range(rank_schoolbook - 1, rank_schoolbook - 10, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_alternating_projection(target_rank, max_iter=200, time_limit=15.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: return schoolbook decomposition
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
