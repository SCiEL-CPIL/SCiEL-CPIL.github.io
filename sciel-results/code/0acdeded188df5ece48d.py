import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _strassen_2x2x2_terms():
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
                    
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
                    
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
                    
            add_term(u, v, w)
            
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
            
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _is_half_integer_vector(vec, tol=1e-5):
    doubled = vec * 2
    rounded = np.round(doubled)
    if np.any(np.abs(doubled - rounded) > tol):
        return False
    return True

def _snap_to_half_int(M):
    return np.round(M * 2) / 2.0

def _als_reduce(U, V, W, n, m, p, max_iter=200, tol=1e-10, deadline=None):
    """
    Alternating least squares to refine a rank-r decomposition.
    Returns (U, V, W, err) where err is sum of squared residuals.
    """
    rank = U.shape[1]
    target_flat = np.ones(n*m*p)
    
    def build_M_U(V, W):
        M = np.zeros((n*m*p, n*m*rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    for k in range(p):
                        t_idx = (i*m + j)*p + k
                        M[t_idx, u_idx*rank + r] = V[j*p + k, r] * W[k*n + i, r]
        return M

    def build_M_V(U, W):
        M = np.zeros((n*m*p, m*p*rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t_idx = (i*m + j)*p + k
                        v_idx = j*p + k
                        M[t_idx, v_idx*rank + r] = U[i*m + j, r] * W[k*n + i, r]
        return M

    def build_M_W(U, V):
        M = np.zeros((n*m*p, p*n*rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t_idx = (i*m + j)*p + k
                        w_idx = k*n + i
                        M[t_idx, w_idx*rank + r] = U[i*m + j, r] * V[j*p + k, r]
        return M

    def compute_err(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Update U
        M_U = build_M_U(V, W)
        U_flat, _, _, _ = np.linalg.lstsq(M_U, target_flat, rcond=None)
        U = U_flat.reshape(n*m, rank)
        
        # Update V
        M_V = build_M_V(U, W)
        V_flat, _, _, _ = np.linalg.lstsq(M_V, target_flat, rcond=None)
        V = V_flat.reshape(m*p, rank)
        
        # Update W
        M_W = build_M_W(U, V)
        W_flat, _, _, _ = np.linalg.lstsq(M_W, target_flat, rcond=None)
        W = W_flat.reshape(p*n, rank)
        
        # Check error
        err = compute_err(U, V, W)
        if err < tol:
            break
            
    return U, V, W, err

def _als_with_perturbation(U, V, W, n, m, p, seed, max_iter=150, tol=1e-10, deadline=None):
    """ALS with random perturbation to escape local minima."""
    rng = np.random.RandomState(seed)
    rank = U.shape[1]
    
    # Small perturbation
    U_pert = U + rng.randn(*U.shape) * 0.01
    V_pert = V + rng.randn(*V.shape) * 0.01
    W_pert = W + rng.randn(*W.shape) * 0.01
    
    return _als_reduce(U_pert, V_pert, W_pert, n, m, p, max_iter, tol, deadline)

def _polish(U, V, W, n, m, p, max_iter=300, tol=1e-12, deadline=None):
    """High-precision polish pass after initial ALS convergence."""
    return _als_reduce(U, V, W, n, m, p, max_iter, tol, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build rank-65 baseline
    U, V, W = _construct_rank_65(n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook rank 80
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
        
    current_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by dropping terms and running ALS
    # Strategy: try dropping 1, 2, 3 terms at a time
    for drop_count in [1, 2, 3]:
        if time.time() > deadline - 30:
            break
            
        target_rank = current_rank - drop_count
        
        # Generate combinations of terms to drop
        # For efficiency, only try a limited number of combinations
        import itertools
        all_indices = list(range(current_rank))
        
        # For drop_count=1, try all
        # For drop_count=2, try a subset
        # For drop_count=3, try a small subset
        if drop_count == 1:
            combos = [tuple([i]) for i in all_indices]
        elif drop_count == 2:
            combos = list(itertools.combinations(all_indices, 2))
            # Limit to first 50 combos for time
            combos = combos[:50]
        else:
            combos = list(itertools.combinations(all_indices, 3))
            combos = combos[:20]
        
        found_better = False
        for combo in combos:
            if time.time() > deadline - 15:
                break
                
            # Remove the specified columns
            keep_mask = np.ones(current_rank, dtype=bool)
            for idx in combo:
                keep_mask[idx] = False
            keep_indices = np.where(keep_mask)[0]
            
            U_sub = U[:, keep_indices]
            V_sub = V[:, keep_indices]
            W_sub = W[:, keep_indices]
            
            # Try ALS with a few different seeds
            for seed in [42, 123, 456]:
                if time.time() > deadline - 10:
                    break
                    
                U_new, V_new, W_new, err = _als_with_perturbation(
                    U_sub, V_sub, W_sub, n, m, p, seed, 
                    max_iter=100, tol=1e-10, deadline=deadline - 10
                )
                
                if err < 1e-8:
                    # Check if entries are close to half-integers
                    if (_is_half_integer_vector(U_new.flatten(), tol=1e-3) and
                        _is_half_integer_vector(V_new.flatten(), tol=1e-3) and
                        _is_half_integer_vector(W_new.flatten(), tol=1e-3)):
                        
                        # Snap to half-integers
                        U_snap = _snap_to_half_int(U_new)
                        V_snap = _snap_to_half_int(V_new)
                        W_snap = _snap_to_half_int(W_new)
                        
                        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                            # Polish pass for higher precision
                            if time.time() < deadline - 5:
                                U_pol, V_pol, W_pol, err_pol = _polish(
                                    U_snap, V_snap, W_snap, n, m, p,
                                    max_iter=200, tol=1e-12, deadline=deadline - 5
                                )
                                if err_pol < 1e-8:
                                    U_snap = _snap_to_half_int(U_pol)
                                    V_snap = _snap_to_half_int(V_pol)
                                    W_snap = _snap_to_half_int(W_pol)
                                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                                    else:
                                        best_U, best_V, W_snap = U_snap, V_snap, W_snap
                                else:
                                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                            else:
                                best_U, best_V, best_W = U_snap, V_snap, W_snap
                            
                            current_rank = best_U.shape[1]
                            found_better = True
                            break
                # Also try without requiring half-integers (for exploration)
                elif err < 1e-10:
                    # Even if not half-integer, remember it as a candidate
                    pass
        
        if found_better:
            break
    
    # Final check: ensure the best result is valid
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
        # Revert to the original rank-65 baseline
        best_U, best_V, best_W = U, V, W
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
