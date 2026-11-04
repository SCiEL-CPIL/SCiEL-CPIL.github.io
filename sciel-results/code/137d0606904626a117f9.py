import time
import numpy as np
from itertools import product

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
    
    # Build the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = sum(U[idx_a, r] * V[idx_b, r] * W[idx_c, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    # Check zeros
    # Build full tensor
    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                # Check if this is a "one" position
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                is_one = (i == i2 and j == j2 and k == k2)
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                expected = 1.0 if is_one else 0.0
                if abs(val - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
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
    
    # Try to find low-rank decomposition
    # Start from lower ranks, go up
    best_result = None
    
    # First, get schoolbook as fallback
    rank_sb = n * m * p
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
    
    def to_lists(mat):
        # Convert to list of lists with half-integer rounding
        rounded = np.round(np.array(mat) * 2) / 2.0
        return rounded.tolist()
    
    def polish(U, V, W, T, max_iters=100, tol=1e-12):
        """Final polish pass with tighter tolerance."""
        r = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Coordinate descent with SVD refinement
            improved = False
            for idx in range(r):
                T_other = T.copy()
                for rr in range(r):
                    if rr == idx:
                        continue
                    T_other -= np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                
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
                
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                U_new[:, idx] = u_vec
                V_new[:, idx] = v2_vec
                W_new[:, idx] = w_vec
                
                T_recon = np.zeros_like(T)
                for rr in range(r):
                    T_recon += np.einsum('a,b,c->abc', U_new[:,rr], V_new[:,rr], W_new[:,rr])
                err_new = np.linalg.norm(T - T_recon)
                
                T_recon_old = np.zeros_like(T)
                for rr in range(r):
                    T_recon_old += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                err_old = np.linalg.norm(T - T_recon_old)
                
                if err_new < err_old - tol:
                    U, V, W = U_new, V_new, W_new
                    improved = True
            
            if not improved:
                break
        
        return U, V, W
    
    def try_rank(r, time_budget):
        """Try to find a rank-r decomposition using random restarts + ALS with structured init."""
        best_local = None
        max_iters = 500
        
        # Multiple initialization strategies
        init_strategies = [
            'random', 'structured', 'block'
        ]
        
        for restart in range(30):
            if time.time() > time_budget:
                break
            try:
                strategy = init_strategies[restart % len(init_strategies)]
                
                if strategy == 'random':
                    U = np.random.randn(nm, r) * 0.3
                    V = np.random.randn(mp, r) * 0.3
                    W = np.random.randn(np_, r) * 0.3
                elif strategy == 'structured':
                    # Initialize with block structure based on matrix multiplication
                    U = np.zeros((nm, r))
                    V = np.zeros((mp, r))
                    W = np.zeros((np_, r))
                    for rr in range(r):
                        # Assign random but structured indices
                        i_idx = np.random.randint(n)
                        j_idx = np.random.randint(m)
                        k_idx = np.random.randint(p)
                        U[i*m+j_idx, rr] = 1.0
                        V[j_idx*p+k_idx, rr] = 1.0
                        W[k_idx*n+i_idx, rr] = 1.0
                        # Add some noise
                        U[i*m+j_idx, rr] += np.random.randn() * 0.1
                        V[j_idx*p+k_idx, rr] += np.random.randn() * 0.1
                        W[k_idx*n+i_idx, rr] += np.random.randn() * 0.1
                else:  # block
                    U = np.zeros((nm, r))
                    V = np.zeros((mp, r))
                    W = np.zeros((np_, r))
                    # Divide rank into blocks for each i
                    block_size = r // n
                    for i in range(n):
                        start = i * block_size
                        end = (i+1) * block_size if i < n-1 else r
                        for rr in range(start, end):
                            j_idx = np.random.randint(m)
                            k_idx = np.random.randint(p)
                            U[i*m+j_idx, rr] = 1.0
                            V[j_idx*p+k_idx, rr] = 1.0
                            W[k_idx*n+i, rr] = 1.0
                
                prev_err = np.inf
                for it in range(max_iters):
                    if time.time() > time_budget:
                        break
                    
                    # Coordinate descent with SVD
                    for idx in range(r):
                        T_other = T.copy()
                        for rr in range(r):
                            if rr == idx:
                                continue
                            T_other -= np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                        
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
                    
                    # Check error
                    T_recon = np.zeros_like(T)
                    for rr in range(r):
                        T_recon += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                    err = np.linalg.norm(T - T_recon)
                    
                    if err < 1e-10:
                        # Polish pass
                        U, V, W = polish(U, V, W, T, max_iters=50, tol=1e-14)
                        T_recon = np.zeros_like(T)
                        for rr in range(r):
                            T_recon += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                        if np.linalg.norm(T - T_recon) < 1e-10:
                            best_local = (U.copy(), V.copy(), W.copy())
                            return best_local
                    if err > prev_err * (1 + 1e-6):
                        break
                    prev_err = err
                    
            except Exception:
                continue
        
        return best_local
    
    # Try ranks from low to high
    min_rank = max(10, n * p)  # lower bound
    for r in range(min_rank, rank_sb + 1):
        if time.time() > deadline - 20:
            break
        time_budget = min(deadline - 15, time.time() + 30)
        result = try_rank(r, time_budget)
        if result is not None:
            U, V, W = result
            # Snap to half-integers
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            
            # Verify
            if _check_decomposition(U_s, V_s, W_s, n, m, p):
                best_result = (U_s, V_s, W_s)
                break
        
        # Also try with the non-snapped version
        if result is not None and best_result is None:
            U, V, W = result
            # Check if already half-integer
            if np.allclose(np.round(U * 2) / 2.0, U, atol=1e-9) and \
               np.allclose(np.round(V * 2) / 2.0, V, atol=1e-9) and \
               np.allclose(np.round(W * 2) / 2.0, W, atol=1e-9):
                best_result = (U, V, W)
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
