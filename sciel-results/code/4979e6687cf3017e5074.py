"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_target(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    result = np.zeros_like(T_a)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T_a, atol=tol)

def _build_rank18(n, m, p):
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
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _als_solve_U(T, V, W, R):
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    
    U = np.zeros((T_a.shape[0], R), dtype=np.float64)
    for u in range(T_a.shape[0]):
        b = T_a[u].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, R):
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    
    V = np.zeros((T_a.shape[1], R), dtype=np.float64)
    for v in range(T_a.shape[1]):
        b = T_a[:, v, :].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    
    W = np.zeros((T_a.shape[2], R), dtype=np.float64)
    for w in range(T_a.shape[2]):
        b = T_a[:, :, w].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _try_rank_als_from_init(U_init, V_init, W_init, T, R, max_iters=200, tol=1e-13):
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, R)
        V = _als_solve_V(T_a, U, W, R)
        W = _als_solve_W(T_a, U, V, R)
        
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
        err = np.sqrt(err / (T_a.size))
        
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

def _try_rank_search(n, m, p, T, R, deadline):
    """Try to find a rank-R decomposition using ALS from various initializations."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: remove each schoolbook term and perturb
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed)
            U_pert = U_init + np.random.randn(4, R) * 0.1
            V_pert = V_init + np.random.randn(10, R) * 0.1
            W_pert = W_init + np.random.randn(10, R) * 0.1
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, R, max_iters=150)
            if result is not None:
                return result
    
    # Strategy 2: remove Strassen terms and perturb
    for term_to_remove in range(14):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(5):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed + 50)
            U_pert = U_init + np.random.randn(4, R) * 0.1
            V_pert = V_init + np.random.randn(10, R) * 0.1
            W_pert = W_init + np.random.randn(10, R) * 0.1
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, R, max_iters=150)
            if result is not None:
                return result
    
    # Strategy 3: random initializations
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(seed * 7 + 123)
        U_init = np.random.randn(4, R) * 0.5
        V_init = np.random.randn(10, R) * 0.5
        W_init = np.random.randn(10, R) * 0.5
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, R, max_iters=100)
        if result is not None:
            return result
    
    # Strategy 4: sparse random initializations
    for seed in range(20):
        if time.time() > deadline:
            break
        np.random.seed(seed * 13 + 456)
        U_init = np.zeros((4, R))
        V_init = np.zeros((10, R))
        W_init = np.zeros((10, R))
        
        for _ in range(20):
            u_idx = np.random.randint(0, 4)
            r_idx = np.random.randint(0, R)
            U_init[u_idx, r_idx] = np.random.choice([-1, 1])
            
        for _ in range(30):
            v_idx = np.random.randint(0, 10)
            r_idx = np.random.randint(0, R)
            V_init[v_idx, r_idx] = np.random.choice([-1, 1])
            
        for _ in range(30):
            w_idx = np.random.randint(0, 10)
            r_idx = np.random.randint(0, R)
            W_init[w_idx, r_idx] = np.random.choice([-1, 1])
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, R, max_iters=100)
        if result is not None:
            return result
    
    return None

def _try_rank16_search(n, m, p, T, deadline):
    """Try to find a rank-16 decomposition."""
    return _try_rank_search(n, m, p, T, 16, deadline)

def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition."""
    return _try_rank_search(n, m, p, T, 17, deadline)

def _try_rank15_search(n, m, p, T, deadline):
    """Try to find a rank-15 decomposition."""
    return _try_rank_search(n, m, p, T, 15, deadline)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    T = _build_target(n, m, p)
    
    # Try to find rank 15 first (best case)
    result = _try_rank15_search(n, m, p, T, deadline - 20)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank 16
    if time.time() < deadline - 20:
        result = _try_rank16_search(n, m, p, T, deadline - 20)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try to find rank 17
    if time.time() < deadline - 20:
        result = _try_rank17_search(n, m, p, T, deadline - 20)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 18
    U, V, W = _build_rank18(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
