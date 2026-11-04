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
    
    # Check zeros
    # We need to check all (a,b,c) that are not ones
    # To be efficient, we can reconstruct the tensor and compare
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

    def try_rank(r, time_budget):
        """Try to find a rank-r decomposition using random restarts + ALS."""
        max_iters = 1000
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Random init
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(np_, r) * 0.5
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > time_budget:
                    break
                
                # ALS Updates
                # Update U: U @ (V[b,r] * W[c,r]) = T[:,b,c]
                # For each b, c, solve least squares? No, U is shared.
                # We want to minimize || T - sum_r U[:,r] (V[:,r] o W[:,r]) ||^2
                # This is not standard ALS.
                
                # Let's use the gradient descent approach on the flattened tensor?
                # Or use the SVD trick from the seed but optimized.
                
                # SVD Trick:
                # For each r, update U[:,r], V[:,r], W[:,r] to best fit the residual.
                # Residual R = T - sum_{rr != r} U[:,rr] V[:,rr] W[:,rr]
                # We want U[:,r] o V[:,r] o W[:,r] ~ R
                # This is equivalent to finding the best rank-1 approximation of R.
                
                # Compute residual for each r? That's O(r^2 * size).
                # Instead, we can do:
                # 1. Compute T_recon = sum_r U[:,r] V[:,r] W[:,r]
                # 2. Residual = T - T_recon
                # 3. For each r, update using SVD of Residual? No, that's greedy.
                
                # Better: Alternating Least Squares on the "unrolled" tensor?
                # Let's stick to the SVD greedy update but optimize it.
                
                # Compute current reconstruction
                T_recon = np.zeros_like(T)
                for rr in range(r):
                    T_recon += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                
                err = np.linalg.norm(T - T_recon)
                if err < 1e-9:
                    return U.copy(), V.copy(), W.copy()
                
                if err > prev_err * (1 + 1e-6):
                    break
                prev_err = err
                
                # Update one column
                idx = it % r
                T_other = T - T_recon + np.einsum('a,b,c->abc', U[:,idx], V[:,idx], W[:,idx])
                
                # SVD of T_other (flattened)
                # T_other is (nm, mp, np_)
                # Reshape to (nm, mp*np_)
                Tf = T_other.reshape(nm, mp*np_)
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

        return None

    # Try ranks from low to high
    # Lower bound is max(nm, mp, np_) = 20
    min_rank = max(nm, mp, np_)
    
    best_result = None
    for r in range(min_rank, rank_sb + 1):
        if time.time() > deadline - 10:
            break
        
        time_budget = min(deadline - 5, time.time() + 20)
        result = try_rank(r, time_budget)
        
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
