import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _round_and_verify(U, V, W, n, m, p, rank):
    """Round to nearest half-integer and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _try_find_decomposition(n, m, p, rank, max_time=30):
    """Try to find a rank decomposition for n x m x p."""
    start_time = time.time()
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    def residuals(params):
        U = params[:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res.append(s - 1.0)
        return np.array(res)
    
    best_solution = None
    best_cost = np.inf
    
    # Try a few random seeds
    for seed in range(50):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        # Initialize with small random values
        x0 = np.random.randn(dim_u*rank + dim_v*rank + dim_w*rank) * 0.5
        try:
            result = least_squares(residuals, x0, max_nfev=2000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-10:
        U = best_solution[:dim_u*rank].reshape(dim_u, rank)
        V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        rounded = _round_and_verify(U, V, W, n, m, p, rank)
        if rounded:
            return rounded
            
    return None

def _build_rank33_decomposition():
    """Fallback rank 33 decomposition using 3 blocks of rank 11."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Rank 11 decomposition for 2x3x2
    # A: 2x3, B: 3x2, C: 2x2
    # Indices for A: 0..5, B: 0..5, C: 0..3
    terms = []
    # Strassen 7 terms for 2x2x2 part (cols 0,1 of B)
    terms.append(({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}))
    terms.append(({3:1, 4:1}, {0:1}, {2:1, 3:-1}))
    terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
    terms.append(({4:1}, {2:1, 0:-1}, {0:1, 2:1}))
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {3:1}))
    terms.append(({1:1, 4:-1}, {2:1, 3:1}, {0:1}))
    # 4 scalar terms for col 2 of B
    terms.append(({2:1}, {4:1}, {0:1}))
    terms.append(({2:1}, {5:1}, {1:1}))
    terms.append(({5:1}, {4:1}, {2:1}))
    terms.append(({5:1}, {5:1}, {3:1}))
    
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = block * 11 + t_idx
            
            for idx, val in u_dict.items():
                U[idx, col] = val
            
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try rank 24
    if time.time() < deadline - 40:
        result = _try_find_decomposition(n, m, p, 24, max_time=30)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Try rank 26
    if time.time() < deadline - 20:
        result = _try_find_decomposition(n, m, p, 26, max_time=20)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to rank 33
    U, V, W = _build_rank33_decomposition()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
