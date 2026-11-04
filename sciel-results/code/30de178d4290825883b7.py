# EVOLVE-BLOCK-START
import numpy as np

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}}
    ]
    
    for j_block in range(4):
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + lv][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[lu * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + (2 + lv)][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[(2 + lu) * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct C using the decomposition: C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    C_reconstructed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        # C[i,k] += sum_j u_r[i,j] * v_r[j,k] * w_r[k,i]
        for i in range(n):
            for k in range(p):
                C_reconstructed[i, k] += w_r[k, i] * np.dot(u_r[i, :], v_r[:, k])
    
    return np.allclose(C_expected, C_reconstructed, atol=1e-9)

def _als_decomposition(target_rank, n_iter=200, seed=0, init=None):
    """ALS for tensor decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    U_size = n * m  # 12
    V_size = m * p  # 32
    W_size = p * n  # 24
    
    # Precompute index arrays
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    idx_u = np.array([i * m + j for i, j, k in indices])
    idx_v = np.array([j * p + k for i, j, k in indices])
    idx_w = np.array([k * n + i for i, j, k in indices])
    
    # Initialize
    if init is not None:
        U, V, W = init
    else:
        U = rng.randn(U_size, target_rank) * 0.5
        V = rng.randn(V_size, target_rank) * 0.5
        W = rng.randn(W_size, target_rank) * 0.5
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(n_iter):
        if time.time() > _START_TIME + _DEADLINE - _MARGIN:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                V_block = V[j * p:(j + 1) * p, :]
                W_vec = np.array([W[k * n + i, :] for k in range(p)])
                P = V_block * W_vec
                try:
                    u, _, _, _ = np.linalg.lstsq(P, np.ones(p), rcond=None)
                    U[i * m + j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                U_block = np.array([U[i * m + j, :] for i in range(n)])
                W_vec = W[k * n:(k + 1) * n, :]
                P = U_block * W_vec
                try:
                    v, _, _, _ = np.linalg.lstsq(P, np.ones(n), rcond=None)
                    V[j * p + k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_block = np.array([U[i * m + j, :] for j in range(m)])
                V_vec = np.array([V[j * p + k, :] for j in range(m)])
                P = U_block * V_vec
                try:
                    w, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                    W[k * n + i, :] = w
                except:
                    pass
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            break
    
    if best_error < 1e-8:
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        
        error_r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    error_r += (val - 1.0) ** 2
        
        if error_r < 1e-8:
            return U_r.tolist(), V_r.tolist(), W_r.tolist(), error_r
        
        U_i = np.round(best_U).astype(float)
        V_i = np.round(best_V).astype(float)
        W_i = np.round(best_W).astype(float)
        error_i = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_i[i*m+j, :], V_i[j*p+k, :] * W_i[k*n+i, :])
                    error_i += (val - 1.0) ** 2
        if error_i < 1e-8:
            return U_i.tolist(), V_i.tolist(), W_i.tolist(), error_i
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    rank88_result = _build_rank88()
    
    # Try to find a better solution
    for target_rank in [40, 45, 50, 60, 70, 80]:
        time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
        if time_remaining < 15:
            break
        
        max_iters = min(100, int(time_remaining * 2))
        for seed in range(2):
            time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
            if time_remaining < 10:
                break
            
            result = _als_decomposition(target_rank, n_iter=max_iters, seed=seed)
            if result is not None:
                U, V, W, err = result
                if _verify_decomposition(U, V, W):
                    return {"U": U, "V": V, "W": W}
    
    return rank88_result
# EVOLVE-BLOCK-END
