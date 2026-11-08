import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _build_tensor():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T, nm, mp, pn

def _als_decompose(T, R, max_iter=500, tol=1e-10, seed=0):
    """Alternating Least Squares for rank-R tensor decomposition."""
    np.random.seed(seed)
    nm, mp, pn = T.shape
    
    U = np.random.randn(nm, R) * 0.01
    V = np.random.randn(mp, R) * 0.01
    W = np.random.randn(pn, R) * 0.01
    
    for it in range(max_iter):
        # Precompute sums for efficient residual computation
        # Residual for U: for each r, we need T - sum_{r'!=r} U[r']*V[r']^T*W[r']
        # This can be written as: for each r, minimize ||T - U[r]*V[r]^T*W[r] - (T - sum_all)||
        
        # Compute full sum
        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
        
        # Update U
        for r in range(R):
            # Residual excluding term r
            # T = S - U[r]*V[r]^T*W[r] + U[r]*V[r]^T*W[r]
            # We want U[r] such that U[r]*V[r]^T*W[r] ≈ T - (S - U[r]*V[r]^T*W[r])
            # = T - S + U[r]*V[r]^T*W[r]
            # Let E = T - S + U[r]*V[r]^T*W[r] (residual with current U[r])
            # But S already includes U[r]*V[r]^T*W[r], so T - S = T - sum_all
            # Actually: S = sum_{r'} U[r']*V[r']^T*W[r']
            # We want: T ≈ S_new where S_new = S - U[r]*V[r]^T*W[r] + U_new[r]*V[r]^T*W[r]
            # So U_new[r]*V[r]^T*W[r] ≈ T - S + U[r]*V[r]^T*W[r]
            # = T - (S - U[r]*V[r]^T*W[r])
            
            E = T - S + np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
            # Now solve for U_new[r]: minimize ||E - U_new[r]*V[r]^T*W[r]||
            # This is: minimize ||E - einsum('ar,br,cr->abc', U_new, V[:,r], W[:,r])||
            # Flatten: E_flat (nm*mp*pn) = M * U_new where M[a,b,c] = V[a,r]*W[b,r]... 
            # Actually the map is: U_new -> U_new[a] * V[b,r] * W[c,r] for all a,b,c
            # This is linear in U_new. The "matrix" is: M[a, (b,c)] = V[b,r]*W[c,r]
            # So E_flat = M @ U_new where E_flat has length nm*mp*pn and U_new has length nm
            # M has shape (nm*mp*pn, nm)
            # M[a*mp*pn + b*pn + c, a'] = V[b,r]*W[c,r] if a==a' else 0
            # This is a block diagonal structure!
            # For each a, the block is: (V[:,r]*W[:,r]) outer product, shape (mp*pn, 1)
            # So it's: for each a, E[a,:,:].flatten() = (V[:,r]*W[:,r]) @ U_new[a]
            # => U_new[a] = <E[a,:,:], V[:,r]*W[:,r]> / ||V[:,r]*W[:,r]||^2
            
            VWr = V[:, r] * W[:, r]  # shape (mp*pn,) ... no, V has shape (mp,R), W has shape (pn,R)
            # V[:,r] has shape (mp,), W[:,r] has shape (pn,)
            # The product V[b,r]*W[c,r] for all b,c is outer product, shape (mp, pn)
            VW_outer = np.outer(V[:, r], W[:, r])  # shape (mp, pn)
            
            U_new = np.zeros(nm)
            for a in range(nm):
                denom = np.sum(VW_outer**2)
                if denom > 1e-15:
                    U_new[a] = np.sum(E[a, :, :] * VW_outer) / denom
                else:
                    U_new[a] = U[a, r]
            U[:, r] = U_new
        
        # Recompute S
        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
        
        # Update V
        for r in range(R):
            E = T - S + np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
            UW_outer = np.outer(U[:, r], W[:, r])  # shape (nm, pn)
            V_new = np.zeros(mp)
            for b in range(mp):
                denom = np.sum(UW_outer**2)
                if denom > 1e-15:
                    V_new[b] = np.sum(E[:, b, :] * UW_outer) / denom
                else:
                    V_new[b] = V[b, r]
            V[:, r] = V_new
        
        # Recompute S
        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
        
        # Update W
        for r in range(R):
            E = T - S + np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
            UV_outer = np.outer(U[:, r], V[:, r])  # shape (nm, mp)
            W_new = np.zeros(pn)
            for c in range(pn):
                denom = np.sum(UV_outer**2)
                if denom > 1e-15:
                    W_new[c] = np.sum(E[:, :, c] * UV_outer) / denom
                else:
                    W_new[c] = W[c, r]
            W[:, r] = W_new
        
        # Check convergence
        S_new = np.zeros_like(T)
        for r in range(R):
            S_new += np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
        err = np.linalg.norm(S_new - T)
        if err < tol:
            break
    
    return U, V, W, err

def _round_to_half_int(arr, tol=1e-6):
    """Round array to nearest half-integer, return (rounded, max_error)."""
    rounded = np.round(arr * 2) / 2
    max_err = np.max(np.abs(arr - rounded))
    return rounded, max_err

def _verify(U, V, W, T, tol=1e-6):
    """Verify that the decomposition exactly reconstructs T."""
    R = U.shape[1]
    S = np.zeros_like(T)
    for r in range(R):
        S += np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
    return np.max(np.abs(S - T)) < tol

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    T, nm, mp, pn = _build_tensor()
    
    # Schoolbook decomposition (rank 36)
    def schoolbook():
        U = np.zeros((nm, 36))
        V = np.zeros((mp, 36))
        W = np.zeros((pn, 36))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    U[i*3+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = 36
    
    # Try to find lower rank decompositions
    # Lower bound is 21. Try ranks from 21 upward.
    for R in range(21, 36):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random initializations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            seed = attempt * 1000 + 42
            U, V, W, err = _als_decompose(T, R, max_iter=300, tol=1e-12, seed=seed)
            
            if err < 1e-8:
                # Check if we can round to half-integers
                U_r, err_u = _round_to_half_int(U)
                V_r, err_v = _round_to_half_int(V)
                W_r, err_w = _round_to_half_int(W)
                
                if err_u < 1e-6 and err_v < 1e-6 and err_w < 1e-6:
                    if _verify(U_r, V_r, W_r, T, tol=1e-6):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = R
                        found = True
                        break
                
                # Also try without rounding (check if exact float works)
                if _verify(U, V, W, T, tol=1e-8):
                    # Try to find rational approximation
                    # For now, accept if it's very close
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    # Return the best result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

