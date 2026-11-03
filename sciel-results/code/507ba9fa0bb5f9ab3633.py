"""Rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        est = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(est - target)) < tol
    
    def try_structured_rank(r, seed=0):
        """
        Try to find a rank-r decomposition using structured initialization.
        Strategy: start from schoolbook and try to merge terms.
        """
        if r >= n * m * p:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.standard_normal((n*m, r))
        V = rng.standard_normal((m*p, r))
        W = rng.standard_normal((p*n, r))
        
        # Use alternating least squares with exact verification
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for it in range(500):
            if time.time() > deadline - 10:
                break
            
            # Compute current estimate
            est = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(est - target))
            
            if err < 1e-9:
                return U, V, W
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Alternating least squares: update one factor at a time
            # Update U: for each r, solve U[:,r] = target * (V[r,:] ⊗ W[r,:])
            # This is a linear system
            for r_idx in range(r):
                # Compute the "slice" of target for this rank component
                # We want: sum_r U[:,r] V[r,:] W[r,:] = target
                # Fix V and W, solve for U[:,r_idx]
                
                # Build the right-hand side: target contracted with V[r_idx,:] and W[r_idx,:]
                # Actually, for ALS: fix all other rank components, solve for one factor
                # est_other = sum_{r' != r_idx} U[:,r'] V[r',:] W[r',:]
                est_other = np.einsum('ar,br,cr->abc', U, V, W)
                est_other[:, :, :] -= np.outer(np.outer(U[:, r_idx], V[r_idx, :]), W[r_idx, :])
                
                resid = target - est_other
                # Now solve: U[:,r_idx] ⊗ (V[r_idx,:] ⊗ W[r_idx,:]) ≈ resid
                # This is a linear system in U[:,r_idx]
                # resid[a,b,c] = U[a,r_idx] * V[b,r_idx] * W[c,r_idx]
                # So U[a,r_idx] = resid[a,b,c] / (V[b,r_idx] * W[c,r_idx])
                # But this must hold for all b,c, so we do a least squares fit
                
                # Build the design matrix: for each a, we have equations:
                # resid[a,b,c] = U[a,r_idx] * V[b,r_idx] * W[c,r_idx]
                # for all b,c
                # So U[a,r_idx] = sum_{b,c} resid[a,b,c] * V[b,r_idx] * W[c,r_idx] / sum_{b,c} (V[b,r_idx] * W[c,r_idx])^2
                
                denom = np.sum(V[r_idx, :] * W[r_idx, :])  # This is wrong, let me reconsider
                
                # Actually: sum_{b,c} (V[b,r_idx] * W[c,r_idx])^2 = (sum_b V[b,r_idx]^2) * (sum_c W[c,r_idx]^2)
                # And the numerator for U[a,r_idx] is: sum_{b,c} resid[a,b,c] * V[b,r_idx] * W[c,r_idx]
                
                V_norm = np.sum(V[r_idx, :] ** 2)
                W_norm = np.sum(W[r_idx, :] ** 2)
                denom = V_norm * W_norm
                
                if denom < 1e-12:
                    continue
                
                # Compute U[:,r_idx]
                # U[a,r_idx] = sum_{b,c} resid[a,b,c] * V[b,r_idx] * W[c,r_idx] / denom
                # This can be computed as: (resid * V[r_idx][:,None,None] * W[r_idx][None,:,None]).sum(axis=(1,2)) / denom
                
                U[:, r_idx] = (resid * V[r_idx, :][:, None, None] * W[r_idx, :][None, :, None]).sum(axis=(1, 2)) / denom
                
                # Similarly update V and W
                est_other = np.einsum('ar,br,cr->abc', U, V, W)
                est_other[:, :, :] -= np.outer(np.outer(U[:, r_idx], V[r_idx, :]), W[r_idx, :])
                resid = target - est_other
                V_norm = np.sum(U[:, r_idx] ** 2)
                W_norm = np.sum(W[r_idx, :] ** 2)
                denom = V_norm * W_norm
                if denom > 1e-12:
                    V[r_idx, :] = (resid * U[:, r_idx][:, None, None] * W[r_idx, :][None, :, None]).sum(axis=(0, 2)) / denom
                
                est_other = np.einsum('ar,br,cr->abc', U, V, W)
                est_other[:, :, :] -= np.outer(np.outer(U[:, r_idx], V[r_idx, :]), W[r_idx, :])
                resid = target - est_other
                U_norm = np.sum(U[:, r_idx] ** 2)
                V_norm = np.sum(V[r_idx, :] ** 2)
                denom = U_norm * V_norm
                if denom > 1e-12:
                    W[r_idx, :] = (resid * U[:, r_idx][:, None, None] * V[r_idx, :][None, :, None]).sum(axis=(0, 1)) / denom
        
        return (best_U, best_V, best_W) if best_err < 1e-9 else None
    
    # Try structured ALS for various ranks
    for r in range(75, 60, -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_structured_rank(r, seed)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 80
    best_rank = n * m * p  # 80
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
