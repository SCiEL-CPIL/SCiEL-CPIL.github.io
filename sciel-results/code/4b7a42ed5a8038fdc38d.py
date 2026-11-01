"""Combined: rank-18 base with aggressive rank-16/17 ALS search."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _build_rank18(n, m, p):
    """Build the known rank-18 Strassen+schoolbook decomposition."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _als_solve_U(T, V, W, n, m, p, R):
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    
    U = np.zeros((n*m, R), dtype=np.float64)
    for u in range(n*m):
        b = T_a[u].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    
    V = np.zeros((m*p, R), dtype=np.float64)
    for v in range(m*p):
        b = T_a[:, v, :].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    
    W = np.zeros((p*n, R), dtype=np.float64)
    for w in range(p*n):
        b = T_a[:, :, w].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=300, tol=1e-12):
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    if _check_decomposition(U, V, W, T_a):
        for denom in [2, 4, 8, 16, 32, 64]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _search_rank(n, m, p, T, R, deadline, U18, V18, W18):
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: Remove terms from rank 18 to get to rank R
    if R < 18:
        terms_to_remove = 18 - R
        # Try removing the last 'terms_to_remove' schoolbook terms first
        # Then try random removals
        indices_to_keep = list(range(18))
        
        # Prioritize removing schoolbook terms (14-17)
        schoolbook_indices = [14, 15, 16, 17]
        strassen_indices = list(range(14))
        
        # Generate combinations of indices to keep
        # For R=16, remove 2. For R=17, remove 1.
        
        # Specific heuristic: remove schoolbook terms first
        if terms_to_remove <= len(schoolbook_indices):
            keep_indices = [i for i in range(18) if i not in schoolbook_indices[:terms_to_remove]]
            U_init = U18_a[:, keep_indices].copy()
            V_init = V18_a[:, keep_indices].copy()
            W_init = W18_a[:, keep_indices].copy()
            
            # Perturb and run ALS
            for seed in range(20):
                if time.time() > deadline:
                    return None
                np.random.seed(seed + 1000)
                scale = 0.1
                U_pert = U_init + np.random.randn(4, R) * scale
                V_pert = V_init + np.random.randn(10, R) * scale
                W_pert = W_init + np.random.randn(10, R) * scale
                
                result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, R, max_iters=200)
                if result is not None:
                    return result

        # Try other combinations of removals
        # Randomly select indices to keep
        for seed in range(50):
            if time.time() > deadline:
                return None
            np.random.seed(seed + 2000)
            keep_indices = np.random.choice(18, R, replace=False)
            U_init = U18_a[:, keep_indices].copy()
            V_init = V18_a[:, keep_indices].copy()
            W_init = W18_a[:, keep_indices].copy()
            
            scale = 0.1
            U_pert = U_init + np.random.randn(4, R) * scale
            V_pert = V_init + np.random.randn(10, R) * scale
            W_pert = W_init + np.random.randn(10, R) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, R, max_iters=200)
            if result is not None:
                return result

    # Strategy 2: Random initialization
    for seed in range(100):
        if time.time() > deadline:
            return None
        np.random.seed(seed + 3000)
        scale = 1.0
        U_init = np.random.randn(4, R) * scale
        V_init = np.random.randn(10, R) * scale
        W_init = np.random.randn(10, R) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=150)
        if result is not None:
            return result
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_target()
    
    # First, build the known rank-18 decomposition
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Verify rank-18
    if not _check_decomposition(U18, V18, W18, T):
        # This should not happen, but just in case
        return {"U": U18, "V": V18, "W": W18}

    # Try to find rank-16
    result16 = _search_rank(n, m, p, T, 16, deadline, U18, V18, W18)
    if result16 is not None:
        U, V, W = result16
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    # Try to find rank-17
    result17 = _search_rank(n, m, p, T, 17, deadline, U18, V18, W18)
    if result17 is not None:
        U, V, W = result17
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    # Fallback: return rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
