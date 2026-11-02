"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Build the multiplication tensor T of shape (p*n, n*m, m*p)
    # T[(k*n+i), (i*m+j), (j*p+k)] = 1
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    
    def verify(U_mat, V_mat, W_mat, atol=1e-10):
        """Verify that sum_r W[:,r] ⊗ U[:,r] ⊗ V[:,r] == T"""
        # T_recon[a,b,c] = sum_r W[a,r] * U[b,r] * V[c,r]
        T_recon = np.einsum('ar,br,cr->abc', W_mat, U_mat, V_mat)
        return np.allclose(T_recon, T, atol=atol)
    
    def try_rank(R, max_restarts=50, max_iters_per_restart=200):
        """Try to find exact rank-R decomposition using iterative refinement."""
        if R >= best_rank:
            return None
        
        # Parameterize: U (n*m x R), V (m*p x R), W (p*n x R)
        # Objective: minimize ||sum_r W_r ⊗ U_r ⊗ V_r - T||_F^2
        # Use gradient-free approach: iteratively optimize one factor at a time
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_r = np.random.randn(n * m, R)
            V_r = np.random.randn(m * p, R)
            W_r = np.random.randn(p * n, R)
            
            # Normalize columns to avoid scale issues
            for r in range(R):
                nu = np.linalg.norm(U_r[:, r])
                nv = np.linalg.norm(V_r[:, r])
                nw = np.linalg.norm(W_r[:, r])
                if nu > 1e-15:
                    U_r[:, r] /= nu
                if nv > 1e-15:
                    V_r[:, r] /= nv
                if nw > 1e-15:
                    W_r[:, r] /= nw
            
            prev_loss = np.inf
            converged = False
            
            for it in range(max_iters_per_restart):
                if time.time() > deadline - 10:
                    break
                
                # Optimize W: for each a, W[a,r] = sum_{b,c} T[a,b,c] * U[b,r] * V[c,r] / sum_r' (U[b,r']*V[c,r']^2)
                # Actually, fix U and V, solve for W:
                # T[a,b,c] = sum_r W[a,r] * U[b,r] * V[c,r]
                # For fixed b,c: T[a,b,c] = sum_r W[a,r] * (U[b,r]*V[c,r])
                # This is: T[:,b,c] = W @ (U[:,b]*V[:,c])
                # So W = T[:,b,c] @ inv(U[:,b]*V[:,c]) for each (b,c), but we need consistency
                # Better: use least squares. For each a:
                # T[a,b,c] for all b,c gives: vec(T[a,:,:]) = (V ⊙ U)^T W[a,:]
                # where ⊙ is element-wise outer product
                # So W[a,:] = vec(T[a,:,:]) @ ((V⊙U)^T)^+
                
                # Compute V⊙U: shape (m*p * n*m, R)
                # (V⊙U)[b,c,r] = V[c,r] * U[b,r]
                VO = np.einsum('cr,br->bcr', V_r, U_r)  # (m*p, n*m, R)
                VO_flat = VO.reshape(m * p * n * m, R)
                
                # For each a: T[a,:,:] reshaped to (m*p*n*m,)
                # T_flat[a, b, c] = T[a, b, c]
                # We want: T_flat[a, :] = W[a, :] @ VO_flat^T
                # So W[a, :] = T_flat[a, :] @ VO_flat^+
                
                T_flat = T.reshape(p * n, m * p * n * m)
                
                # Pseudo-inverse of VO_flat^T
                # VO_flat is (m*p*n*m, R), so VO_flat^T is (R, m*p*n*m)
                # We need W[a,:] (1 x R) such that W[a,:] @ VO_flat^T = T_flat[a,:]
                # This is underdetermined if R < m*p*n*m, so use least squares
                # W[a,:] = T_flat[a,:] @ (VO_flat^T)^+ = T_flat[a,:] @ VO_flat^+ @ (VO_flat^+)^T... 
                # Actually: minimize ||W[a,:] @ VO_flat^T - T_flat[a,:]||^2
                # Solution: W[a,:] = T_flat[a,:] @ (VO_flat^T)^+
                
                # Compute (VO_flat^T)^+ = (VO_flat^T)^T @ (VO_flat^T @ (VO_flat^T)^T)^+ 
                # = VO_flat @ (VO_flat^T @ VO_flat)^+ ... this is getting complex
                # Use np.linalg.lstsq directly
                
                # Actually, for each a, solve: VO_flat^T @ W[a,:]^T = T_flat[a,:]^T
                # i.e., VO_flat^T (R x m*p*n*m) @ W[a,:]^T (R x 1) = T_flat[a,:]^T (m*p*n*m x 1)
                # This is overdetermined, use lstsq
                
                W_new = np.zeros((p * n, R))
                for a in range(p * n):
                    # Solve VO_flat^T @ x = T_flat[a, :]^T
                    # VO_flat^T is (R, m*p*n*m), T_flat[a,:] is (m*p*n*m,)
                    # We want x (R,) such that VO_flat^T @ x = T_flat[a, :]
                    # Use lstsq: x = np.linalg.lstsq(VO_flat^T, T_flat[a,:], rcond=None)[0]
                    x, residuals, rank_s, sv = np.linalg.lstsq(VO_flat.T, T_flat[a, :], rcond=None)
                    W_new[a, :] = x
                
                # Similarly optimize U
                # Fix V and W, solve for U
                # T[a,b,c] = sum_r U[b,r] * (W[a,r] * V[c,r])
                # For fixed a,c: T[a,:,c] = U @ (W[a,:]*V[c,:])^T
                # U[b,:] = T[a,b,c] for all a,c... 
                # vec(T[:,b,c]) for fixed b: shape (p*n*m*p,)
                # = (W ⊙ V)^T @ U[b,:]
                # W⊙V: (p*n, m*p, R), (W⊙V)[a,c,r] = W[a,r]*V[c,r]
                WV = np.einsum('ar,cr->acr', W_new, V_r)  # (p*n, m*p, R)
                WV_flat = WV.reshape(p * n * m * p, R)
                T_flat2 = T.reshape(m * p * n * m, p * n * m * p)  # Hmm, need to be careful with indexing
                
                # Let me re-index: T[a,b,c] where a in [0,p*n), b in [0,n*m), c in [0,m*p)
                # For fixed b: T[:,b,:] has shape (p*n, m*p)
                # vec(T[:,b,:]) = (W⊙V)^T @ U[b,:]
                # W⊙V[a,c,r] = W[a,r]*V[c,r]
                # So (W⊙V)^T has shape (R, p*n*m*p)
                # T[:,b,:] reshaped to (p*n*m*p,)
                
                U_new = np.zeros((n * m, R))
                for b in range(n * m):
                    T_b = T[:, b, :].reshape(p * n * m * p)
                    x, residuals, rank_s, sv = np.linalg.lstsq(WV_flat.T, T_b, rcond=None)
                    U_new[b, :] = x
                
                # Optimize V
                # T[a,b,c] = sum_r V[c,r] * (U[b,r] * W[a,r])
                # For fixed a,b: T[a,b,:] = V @ (U[b,:]*W[a,:])^T
                UW = np.einsum('br,ar->abr', U_new, W_new)  # (n*m, p*n, R)
                UW_flat = UW.reshape(n * m * p * n, R)
                
                V_new = np.zeros((m * p, R))
                for c in range(m * p):
                    T_c = T[:, :, c].reshape(p * n * n * m)  # Wait, T[a,b,c] for fixed c: shape (p*n, n*m)
                    # vec(T[:, :, c]) = (U⊙W)^T @ V[c,:]
                    # U⊙W[b,a,r] = U[b,r]*W[a,r]
                    # (U⊙W)^T has shape (R, n*m*p*n)
                    # T[:, :, c] reshaped to (p*n*n*m) - but ordering matters
                    T_c = T[:, :, c].reshape(-1)
                    x, residuals, rank_s, sv = np.linalg.lstsq(UW_flat.T, T_c, rcond=None)
                    V_new[c, :] = x
                
                U_r, V_r, W_r = U_new, V_new, W_new
                
                # Compute loss
                T_recon = np.einsum('ar,br,cr->abc', W_r, U_r, V_r)
                loss = np.linalg.norm(T_recon - T)
                
                if loss < 1e-10:
                    converged = True
                    break
                
                if abs(prev_loss - loss) < 1e-15:
                    break
                prev_loss = loss
            
            if converged:
                return U_r, V_r, W_r
        
        return None
    
    # Try to find lower rank decompositions
    for R in range(41, 20, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank(R, max_restarts=30, max_iters_per_restart=100)
        if result is not None:
            U, V, W = result
            best_rank = R
            break
    
    # Final verification
    if not verify(U, V, W):
        U, V, W = schoolbook()
        best_rank = n * m * p
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

