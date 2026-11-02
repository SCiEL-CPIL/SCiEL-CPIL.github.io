"""Combined: rank-18 base with enhanced rank-17 search using tensor structure."""
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
    """Solve for U given V, W using least squares."""
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
    """Solve for V given U, W."""
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
    """Solve for W given U, V."""
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

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=200, tol=1e-13):
    """Run ALS from a given initialization, return (U, V, W) as lists if exact."""
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
        # Try to find rational representation
        for denom in [2, 4, 8, 16, 32, 64, 128]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _tensor_svd_analysis(T, n, m, p, R):
    """
    Use tensor flattening and SVD to analyze structure and find good initializations.
    Flatten the tensor in different ways and look at singular values.
    """
    T_a = np.array(T, dtype=np.float64)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Flattening 1: u vs (v,w)
    T_uvw = T_a.reshape(n_m, m_p * p_n)
    U1, S1, Vt1 = np.linalg.svd(T_uvw, full_matrices=False)
    
    # Flattening 2: v vs (u,w)
    T_vuw = T_a.transpose(1, 0, 2).reshape(m_p, n_m * p_n)
    U2, S2, Vt2 = np.linalg.svd(T_vuw, full_matrices=False)
    
    # Flattening 3: w vs (u,v)
    T_wuv = T_a.transpose(2, 0, 1).reshape(p_n, n_m * m_p)
    U3, S3, Vt3 = np.linalg.svd(T_wuv, full_matrices=False)
    
    # The rank of the tensor is at least the max of these matrix ranks
    rank_u = np.sum(S1 > 1e-10)
    rank_v = np.sum(S2 > 1e-10)
    rank_w = np.sum(S3 > 1e-10)
    
    return {
        'rank_u': rank_u,
        'rank_v': rank_v, 
        'rank_w': rank_w,
        'U1': U1, 'S1': S1, 'Vt1': Vt1,
        'U2': U2, 'S2': S2, 'Vt2': Vt2,
        'U3': U3, 'S3': S3, 'Vt3': Vt3
    }

def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition with enhanced strategies including tensor SVD."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 0: Tensor SVD analysis to find good subspaces
    try:
        svd_info = _tensor_svd_analysis(T, n, m, p, 17)
    except:
        svd_info = None
    
    # Strategy 1: remove each schoolbook term and perturb with varying scales
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(15):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed)
            scale = 0.005 + 0.005 * (seed % 7)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=250)
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
        
        for seed in range(8):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed + 50)
            scale = 0.005 + 0.005 * (seed % 7)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=250)
            if result is not None:
                return result
    
    # Strategy 3: Use SVD-informed initializations
    if svd_info is not None:
        for seed in range(20):
            if time.time() > deadline:
                break
            np.random.seed(seed * 13 + 456)
            
            # Try to use the top singular vectors as initial subspace hints
            # This is heuristic - we mix random with SVD-inspired structure
            scale = 0.2 + 0.1 * (seed % 5)
            U_init = np.random.randn(4, 17) * scale
            V_init = np.random.randn(10, 17) * scale
            W_init = np.random.randn(10, 17) * scale
            
            # Occasionally bias toward the SVD structure
            if seed % 4 == 0 and svd_info['U1'].shape[1] >= 17:
                # Use top 17 singular vectors as a hint (heuristic)
                hint = svd_info['U1'][:, :min(17, svd_info['U1'].shape[1])]
                U_init = U_init * 0.5 + np.random.randn(4, 17) * 0.5
            
            result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=200)
            if result is not None:
                return result
    
    # Strategy 4: Random initializations with varying scales
    for seed in range(40):
        if time.time() > deadline:
            break
        np.random.seed(seed * 7 + 123)
        scale = 0.2 + 0.15 * (seed % 6)
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=180)
        if result is not None:
            return result
    
    # Strategy 5: Try merging terms by finding near-linear dependencies
    # Check if any pair of rank-1 terms can be combined
    for i in range(18):
        if time.time() > deadline:
            break
        for j in range(i+1, 18):
            if time.time() > deadline:
                break
            
            # Check if removing both and replacing with 1 term works
            idx = [k for k in range(18) if k != i and k != j]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            # Add one new random term
            np.random.seed(i * 100 + j)
            new_U = np.random.randn(4, 1) * 0.5
            new_V = np.random.randn(10, 1) * 0.5
            new_W = np.random.randn(10, 1) * 0.5
            
            U_full = np.hstack([U_init, new_U])
            V_full = np.hstack([V_init, new_V])
            W_full = np.hstack([W_init, new_W])
            
            result = _try_rank_als_from_init(U_full, V_full, W_full, T, n, m, p, 17, max_iters=150)
            if result is not None:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    T = _build_target()
    
    # First, build the known rank-18 decomposition
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Verify rank-18
    if _check_decomposition(U18, V18, W18, T):
        # Try to find rank-17
        result17 = _try_rank17_search(n, m, p, T, deadline)
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
