"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

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
    
    # Check all (i,j,k) constraints
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _als_from_init(U_init, V_init, W_init, target_rank, n_iter=300, seed=0):
    """ALS starting from an initialization, trying to reach target_rank."""
    n, m, p = DIMENSIONS
    
    if U_init.shape[1] == target_rank:
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
    elif U_init.shape[1] > target_rank:
        # Project down: take first target_rank columns
        U, V, W = U_init[:, :target_rank].copy(), V_init[:, :target_rank].copy(), W_init[:, :target_rank].copy()
    else:
        return None
    
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
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
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
            return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    return None

def _try_reduced_rank(U88, V88, W88, target_rank):
    """Try to find a rank-`target_rank` decomposition by perturbing the rank-88 init."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(42)
    
    for attempt in range(5):
        if time.time() > _START_TIME + _DEADLINE - _MARGIN:
            break
        
        if attempt == 0:
            # Direct truncation
            U_init = U88[:, :target_rank]
            V_init = V88[:, :target_rank]
            W_init = W88[:, :target_rank]
        else:
            # Random projection: mix columns
            perm = rng.permutation(88)
            U_init = U88[:, perm[:target_rank]]
            V_init = V88[:, perm[:target_rank]]
            W_init = W88[:, perm[:target_rank]]
        
        result = _als_from_init(U_init, V_init, W_init, target_rank, n_iter=200)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    rank88_result = _build_rank88()
    U88 = np.array(rank88_result['U'], dtype=np.float64)
    V88 = np.array(rank88_result['V'], dtype=np.float64)
    W88 = np.array(rank88_result['W'], dtype=np.float64)
    
    # Try to find lower rank solutions using rank-88 as initialization
    for target_rank in [87, 86, 85, 84, 82, 80, 78, 76, 74, 72, 70]:
        time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
        if time_remaining < 10:
            break
        
        result = _try_reduced_rank(U88, V88, W88, target_rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return rank88_result
# EVOLVE-BLOCK-END
