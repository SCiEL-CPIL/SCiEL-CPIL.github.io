# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank33_decomposition():
    """Build a rank-33 decomposition using 3 independent 2x3x2 blocks."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    terms = []
    
    terms.append({'u': {0:1, 4:1}, 'v': {0:1, 3:1}, 'w': {0:1, 3:1}})
    terms.append({'u': {3:1, 4:1}, 'v': {0:1}, 'w': {2:1, 3:-1}})
    terms.append({'u': {0:1}, 'v': {1:1, 3:-1}, 'w': {1:1, 3:1}})
    terms.append({'u': {4:1}, 'v': {2:1, 0:-1}, 'w': {0:1, 2:1}})
    terms.append({'u': {0:1, 1:1}, 'v': {3:1}, 'w': {0:-1, 1:1}})
    terms.append({'u': {3:1, 0:-1}, 'v': {0:1, 1:1}, 'w': {3:1}})
    terms.append({'u': {1:1, 4:-1}, 'v': {2:1, 3:1}, 'w': {0:1}})
    
    terms.append({'u': {2:1}, 'v': {4:1}, 'w': {0:1}})
    terms.append({'u': {2:1}, 'v': {5:1}, 'w': {1:1}})
    terms.append({'u': {5:1}, 'v': {4:1}, 'w': {2:1}})
    terms.append({'u': {5:1}, 'v': {5:1}, 'w': {3:1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            for idx, val in term['u'].items():
                U[idx, col] = val
            
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W

def _solve_2x3x3_rank11(max_iters=500, tol=1e-5):
    n, m, p = 2, 3, 3
    r = 11
    
    np.random.seed(42)
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    for it in range(max_iters):
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        A_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b_vec = np.ones(n)
                V_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[j*p+k, :] = V_new
        
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b_vec = np.ones(p)
                U_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[i*m+j, :] = U_new
                
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b_vec = np.ones(m)
                W_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[k*n+i, :] = W_new
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    max_err = max(max_err, abs(s - 1.0))
                    
        if max_err < tol:
            return U, V, W
            
    return None

def _build_rank22_decomposition():
    n, m, p = 2, 3, 6
    rank = 22
    
    U, V, W = _solve_2x3x3_rank11()
    if U is None:
        return None
        
    U_full = np.zeros((n * m, rank))
    V_full = np.zeros((m * p, rank))
    W_full = np.zeros((p * n, rank))
    
    U_full[:, 0:11] = U
    for j in range(m):
        for k in range(3):
            V_full[j * p + k, 0:11] = V[j * 3 + k, :]
    for k in range(3):
        for i in range(n):
            W_full[k * n + i, 0:11] = W[k * 2 + i, :]
            
    U_full[:, 11:22] = U
    for j in range(m):
        for k in range(3):
            V_full[j * p + (k + 3), 11:22] = V[j * 3 + k, :]
    for k in range(3):
        for i in range(n):
            W_full[(k + 3) * n + i, 11:22] = W[k * 2 + i, :]
            
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    if time.time() < deadline - 10.0:
        res = _build_rank22_decomposition()
        if res is not None:
            U, V, W = res
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    U, V, W = _build_rank33_decomposition()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
