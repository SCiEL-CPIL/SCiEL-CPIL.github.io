import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check ones
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zeros by reconstructing full tensor
    T_recon = np.zeros((nm, mp, np_))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    if np.linalg.norm(T_recon - T_target) > tol:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Precompute unfoldings
    T_unfolded_a = T.reshape(nm, mp * np_)  # for U
    T_unfolded_b = T.transpose(1, 0, 2).reshape(mp, nm * np_)  # for V
    T_unfolded_c = T.transpose(2, 0, 1).reshape(np_, nm * mp)  # for W

    # Schoolbook rank
    rank_sb = n * m * p
    
    # Prepare schoolbook fallback
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    def khatri_rao(A, B):
        """Compute Khatri-Rao product of A (rows_a x r) and B (rows_b x r).
        Returns (rows_a*rows_b x r) where column r is outer(A[:,r], B[:,r]).flatten()"""
        r = A.shape[1]
        return np.einsum('ir,jr->ijr', A, B).reshape(A.shape[0] * B.shape[0], r)

    def try_rank_als(r, time_budget, max_restarts=5, max_iters=300):
        """Try to find a rank-r decomposition using ALS."""
        best_err = np.inf
        best_result = None
        
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Random init
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(np_, r) * 0.1
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > time_budget:
                    break
                
                # Update U: T_unfolded_a ≈ U @ khatri_rao(V, W)
                VW = khatri_rao(V, W)
                U_new, _, _, _ = np.linalg.lstsq(VW, T_unfolded_a, rcond=None)
                
                # Update V: T_unfolded_b ≈ V @ khatri_rao(U, W)
                UW = khatri_rao(U, W)
                V_new, _, _, _ = np.linalg.lstsq(UW, T_unfolded_b, rcond=None)
                
                # Update W: T_unfolded_c ≈ W @ khatri_rao(U, V)
                UV = khatri_rao(U, V)
                W_new, _, _, _ = np.linalg.lstsq(UV, T_unfolded_c, rcond=None)
                
                U = U_new
                V = V_new
                W = W_new
                
                # Compute error
                T_recon = np.zeros((nm, mp, np_))
                for rr in range(r):
                    T_recon += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                err = np.linalg.norm(T - T_recon)
                
                if err < 1e-9:
                    return U.copy(), V.copy(), W.copy()
                
                if err > prev_err * (1 + 1e-6) and it > 10:
                    break
                prev_err = err
                
                if err < best_err:
                    best_err = err
                    best_result = (U.copy(), V.copy(), W.copy())
        
        if best_result is not None and best_err < 1e-6:
            return best_result
        return None

    # Try ranks from low to high
    min_rank = max(nm, mp, np_)
    
    best_result = None
    for r in range(min_rank, rank_sb + 1):
        if time.time() > deadline - 15:
            break
        
        time_budget = min(deadline - 5, time.time() + 25)
        result = try_rank_als(r, time_budget)
        
        if result is not None:
            U, V, W = result
            # Snap to half-integers if possible
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            
            if _check_decomposition(U_s, V_s, W_s, n, m, p):
                best_result = (U_s, V_s, W_s)
                break
            elif _check_decomposition(U, V, W, n, m, p):
                best_result = (U, V, W)
                break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
