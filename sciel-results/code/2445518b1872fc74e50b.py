import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140.0

    # Baseline rank 53
    baseline = _build_rank53()
    best_result = baseline
    best_rank = 53

    # Try to find rank 10 for 2x2x3
    if time.time() < deadline - 30:
        rank10_factors = _find_rank10_2x2x3(deadline - 30)
        if rank10_factors is not None:
            # Tile 5 times
            result = _tile_2x2x3(rank10_factors, 5)
            best_result = result
            best_rank = 50

    return best_result

def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 factors (rank 7)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # 2x2x1 factors (last column, rank 4)
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7

    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]

    return {"U": U, "V": V, "W": W}

def _compute_residual_fast(U, V, W):
    n, m, p = 2, 2, 15
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
    return T

def _build_target_tensor_2x2x3():
    n, m, p = 2, 2, 3
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _find_rank10_2x2x3(deadline):
    n, m, p = 2, 2, 3
    target_rank = 10
    T_target = _build_target_tensor_2x2x3()
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Rank 11 solution (Strassen + Naive)
    # Strassen 2x2x2 (rank 7) + Naive 2x2x1 (rank 4)
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0]
    ])
    # V for 2x2x2: m*p = 2*2 = 4 rows.
    # V_s is 4x7.
    # We need V for 2x2x3.
    # Columns 0,1 for Strassen, Column 2 for Naive.
    V_11 = np.zeros((4, 11))
    V_11[:, :7] = V_s
    # Naive for col 2:
    # V[j*3+2, r]
    # j=0, r=0: 1
    # j=1, r=1: 1
    # Other r: 0
    # But we need to map to 11 cols.
    # Let's just use random for simplicity if this is complex.
    
    # Better: Use random initialization with multiple restarts.
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        U0 = rng.normal(0, 1.0, (n*m, target_rank))
        V0 = rng.normal(0, 1.0, (m*p, target_rank))
        W0 = rng.normal(0, 1.0, (p*n, target_rank))
        
        def _flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def _unflatten(x):
            U_size = n * m * target_rank
            V_size = m * p * target_rank
            U = x[:U_size].reshape(n * m, target_rank)
            V = x[U_size:U_size+V_size].reshape(m * p, target_rank)
            W = x[U_size+V_size:].reshape(p * n, target_rank)
            return U, V, W
        
        def _residual(x):
            U, V, W = _unflatten(x)
            # Compute T
            T_comp = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_comp[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            return (T_comp - T_target).flatten()
        
        x0 = _flatten(U0, V0, W0)
        
        try:
            result = least_squares(_residual, x0, method='lm', max_nfev=1000)
            if np.linalg.norm(result.fun) < 1e-8:
                U_opt, V_opt, W_opt = _unflatten(result.x)
                # Round to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify exactness
                T_check = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_check[i, j, k] = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                
                if np.max(np.abs(T_check - T_target)) < 1e-10:
                    return (U_rounded, V_rounded, W_rounded)
        except Exception:
            continue
            
    return None

def _tile_2x2x3(factors, num_tiles):
    n, m, p_block = 2, 2, 3
    U_block, V_block, W_block = factors
    r_block = U_block.shape[1]
    
    total_rank = r_block * num_tiles
    n_full = 2
    m_full = 2
    p_full = 15
    
    U_full = np.zeros((n_full * m_full, total_rank))
    V_full = np.zeros((m_full * p_full, total_rank))
    W_full = np.zeros((p_full * n_full, total_rank))
    
    for t in range(num_tiles):
        k_start = t * p_block
        r_start = t * r_block
        r_end = r_start + r_block
        
        # U is same for all tiles
        U_full[:, r_start:r_end] = U_block
        
        # V
        for j in range(m_full):
            for k in range(p_block):
                V_full[j * p_full + k_start + k, r_start:r_end] = V_block[j * p_block + k, :]
        
        # W
        for k in range(p_block):
            for i in range(n_full):
                W_full[(k_start + k) * n_full + i, r_start:r_end] = W_block[k * n_full + i, :]
                
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
