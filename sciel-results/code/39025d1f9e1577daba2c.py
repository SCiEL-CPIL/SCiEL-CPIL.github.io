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

def _als_reduce(U, V, W, n, m, p, max_iter=100, tol=1e-8):
    """
    Attempts to reduce rank by 1 using ALS.
    """
    rank = U.shape[1]
    target = np.ones((n, m, p))
    
    # Pre-compute indices for efficiency
    # M matrix for U update: shape (n*m*p, n*m*rank)
    # M[(i,j,k), (i',j',r)] = V[j*p+k, r] * W[k*n+i, r] if i==i', j==j' else 0
    # This is sparse. We can use scipy.sparse or just dense if small.
    # n*m*p = 80. n*m*rank = 16*64 = 1024. Dense is fine.
    
    M_U = np.zeros((n*m*p, n*m*rank))
    M_V = np.zeros((n*m*p, m*p*rank))
    M_W = np.zeros((n*m*p, p*n*rank))
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_idx = i*m + j
                for k in range(p):
                    t_idx = (i*m + j)*p + k
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    
                    # U update: U[i*m+j, r] * v_val * w_val
                    M_U[t_idx, u_idx*rank + r] = v_val * w_val
                    
                    # V update: V[j*p+k, r] * u_val * w_val
                    # M_V[t_idx, v_idx*rank + r] = U[i*m+j, r] * W[k*n+i, r]
                    # We will update M_V inside the loop as U changes? 
                    # No, ALS updates one factor at a time.
                    # When updating U, V and W are fixed.
                    # So we build M_U once per iteration? No, M_U depends on V,W.
                    # M_V depends on U,W.
                    # M_W depends on U,V.
                    
    # We need to rebuild M matrices if U,V,W change.
    # But in ALS, we fix two and update one.
    # So:
    # 1. Fix V,W. Build M_U. Solve for U.
    # 2. Fix U,W. Build M_V. Solve for V.
    # 3. Fix U,V. Build M_W. Solve for W.
    
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

    target_flat = target.flatten()
    
    for it in range(max_iter):
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
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            break
            
    return U, V, W, err

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U, V, W = _construct_rank_65(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback
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
    
    # Try to reduce to rank 64
    if time.time() < deadline - 20:
        # Try dropping each term
        for r_drop in range(current_rank):
            if time.time() > deadline - 5:
                break
                
            U_sub = np.delete(U, r_drop, axis=1)
            V_sub = np.delete(V, r_drop, axis=1)
            W_sub = np.delete(W, r_drop, axis=1)
            
            # Run ALS
            U_new, V_new, W_new, err = _als_reduce(U_sub, V_sub, W_sub, n, m, p, max_iter=50)
            
            if err < 1e-8:
                # Check half-integers
                if (_is_half_integer_vector(U_new.flatten(), tol=1e-4) and
                    _is_half_integer_vector(V_new.flatten(), tol=1e-4) and
                    _is_half_integer_vector(W_new.flatten(), tol=1e-4)):
                    
                    # Snap
                    U_snap = np.round(U_new * 2) / 2.0
                    V_snap = np.round(V_new * 2) / 2.0
                    W_snap = np.round(W_new * 2) / 2.0
                    
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                        U, V, W = U_snap, V_snap, W_snap
                        current_rank = U.shape[1]
                        break
                        
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
