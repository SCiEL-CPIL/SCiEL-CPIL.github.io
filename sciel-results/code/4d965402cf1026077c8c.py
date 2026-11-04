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
    
    # Build target tensor
    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct
    T_recon = np.zeros((nm, mp, np_))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
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

    def snap_to_half(x):
        return np.round(x * 2) / 2.0

    def try_reduce_rank(U0, V0, W0, target_rank, time_budget):
        """Try to reduce rank from current to target_rank using structured search."""
        current_rank = U0.shape[1]
        if target_rank >= current_rank:
            return None
        
        # Try to find linear dependencies
        # For each pair of columns, check if they can be merged
        # This is expensive, so we'll use a heuristic
        
        # Alternative: Use a more direct optimization
        # Initialize at target_rank with random init but guided by structure
        
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Initialize with random but structured
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(np_, target_rank) * 0.3
            
            # Add some structure: ensure the one-positions are well-fit
            # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            best_err = np.inf
            for it in range(500):
                if time.time() > time_budget:
                    break
                
                # Compute current error
                T_recon = np.zeros_like(T)
                for r in range(target_rank):
                    T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                err = np.linalg.norm(T - T_recon)
                
                if err < 1e-10:
                    return U.copy(), V.copy(), W.copy()
                
                if err > best_err:
                    break
                best_err = err
                
                # Update one column at a time using SVD
                idx = it % target_rank
                T_other = T - T_recon + np.einsum('a,b,c->abc', U[:,idx], V[:,idx], W[:,idx])
                
                Tf = T_other.reshape(nm, mp*np_)
                try:
                    U_, S_, Vt_ = np.linalg.svd(Tf, full_matrices=False)
                    if S_[0] < 1e-15:
                        continue
                    u_vec = U_[:, 0] * np.sqrt(S_[0])
                    v_vec_full = Vt_[0, :] * np.sqrt(S_[0])
                    V_mat = v_vec_full.reshape(mp, np_)
                    V2_, S2_, Wt_ = np.linalg.svd(V_mat, full_matrices=False)
                    if S2_[0] < 1e-15:
                        continue
                    v2_vec = V2_[:, 0] * np.sqrt(S2_[0])
                    w_vec = Wt_[0, :] * np.sqrt(S2_[0])
                    
                    U[:, idx] = u_vec
                    V[:, idx] = v2_vec
                    W[:, idx] = w_vec
                except Exception:
                    continue
        
        return None

    # Try ranks from lower to higher
    # Start from a reasonable lower bound
    min_rank = max(nm, mp, np_)  # 20
    
    best_result = None
    for r in range(min_rank, rank_sb + 1):
        if time.time() > deadline - 15:
            break
        
        time_budget = min(deadline - 5, time.time() + 15)
        result = try_reduce_rank(U_sb, V_sb, W_sb, r, time_budget)
        
        if result is not None:
            U, V, W = result
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            
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
