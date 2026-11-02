import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify(U, V, W, n=2, m=2, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _build_tensor(n=2, m=2, p=5):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    
    return U, V, W

def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-12):
    prev_err = np.inf
    for it in range(max_iter):
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        err = np.linalg.norm(T - np.einsum('ur,vr,wr->uvw', U, V, W))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    result = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(result, T, atol=1e-6):
        return None
    
    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.einsum('ur,vr,wr->uvw', U_s, V_s, W_s)
        if np.allclose(result_s, T, atol=1e-5):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-8:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _search_rank_aggressive(n, m, p, R, timeout=10, n_trials=200):
    """Aggressive ALS search with multiple initialization strategies."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # Base Strassen block for initialization
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    
    for trial in range(n_trials):
        if time.time() - start_time > timeout:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Strategy 1: Start from Strassen + random completion
        if trial % 3 == 0:
            U[:, :7] = U_str
            V[:, :7] = V_str
            W[:, :7] = W_str
            scale = 0.3 + 0.2 * (trial % 5)
            U[:, 7:] = np.random.randn(n*m, R-7) * scale
            V[:, 7:] = np.random.randn(m*p, R-7) * scale
            W[:, 7:] = np.random.randn(p*n, R-7) * scale
        
        # Strategy 2: Pure random with small scale
        elif trial % 3 == 1:
            scale = 0.5 + 0.1 * (trial % 4)
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
        
        # Strategy 3: Two Strassen blocks + random
        else:
            U[:, :7] = U_str
            V[:, :7] = V_str
            W[:, :7] = W_str
            U_str2, V_str2, W_str2 = _get_strassen_block_np(n, m, p, 2, 3)
            if R >= 14:
                U[:, 7:14] = U_str2
                V[:, 7:14] = V_str2
                W[:, 7:14] = W_str2
            if R > 14:
                U[:, 14:] = np.random.randn(n*m, R-14) * 0.3
                V[:, 14:] = np.random.randn(m*p, R-14) * 0.3
                W[:, 14:] = np.random.randn(p*n, R-14) * 0.3
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=400)
        
        if err < 1e-8:
            result = _check_and_rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    return None

def _construct_rank18(n=2, m=2, p=5):
    """Known valid rank-18 decomposition: two Strassen blocks + schoolbook for col 4."""
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    U_str2, V_str2, W_str2 = _get_strassen_block_np(n, m, p, 2, 3)
    U[:, 7:14] = U_str2
    V[:, 7:14] = V_str2
    W[:, 7:14] = W_str2
    
    # Schoolbook for column 4
    U[0, 14] = 1; V[0*p+4, 14] = 1; W[4*n+0, 14] = 1
    U[1, 15] = 1; V[1*p+4, 15] = 1; W[4*n+0, 15] = 1
    U[2, 16] = 1; V[0*p+4, 16] = 1; W[4*n+1, 16] = 1
    U[3, 17] = 1; V[1*p+4, 17] = 1; W[4*n+1, 17] = 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Verify before returning
    if not _verify(U_list, V_list, W_list, n, m, p):
        raise ValueError("Rank-18 construction failed verification")
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_explicit_rank15(n=2, m=2, p=5):
    """
    Try to find an explicit rank-15 decomposition using algebraic structure.
    The 2x2x5 multiplication tensor has a special structure that may allow
    rank 15 = 7 + 8 or 7 + 7 + 1 or other combinations.
    """
    T = _build_tensor(n, m, p)
    R = 15
    
    # Try with different random seeds and initializations
    for seed in range(50):
        np.random.seed(seed)
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=600)
        
        if err < 1e-10:
            result = _check_and_rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    return None

def _try_explicit_rank16(n=2, m=2, p=5):
    """Try to find an explicit rank-16 decomposition."""
    T = _build_tensor(n, m, p)
    R = 16
    
    for seed in range(50):
        np.random.seed(seed)
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=600)
        
        if err < 1e-10:
            result = _check_and_rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try Rank 15 - most aggressive search
    elapsed = time.time() - START_TIME
    t15_budget = min(40, deadline - time.time() - 35)
    if t15_budget > 5:
        result = _search_rank_aggressive(n, m, p, 15, timeout=t15_budget, n_trials=150)
        if result is not None:
            return result
        # Also try explicit seeded search
        remaining = deadline - time.time() - 30
        if remaining > 5:
            result = _try_explicit_rank15(n, m, p)
            if result is not None:
                return result
    
    # Try Rank 16
    elapsed = time.time() - START_TIME
    t16_budget = min(30, deadline - time.time() - 25)
    if t16_budget > 5:
        result = _search_rank_aggressive(n, m, p, 16, timeout=t16_budget, n_trials=100)
        if result is not None:
            return result
        remaining = deadline - time.time() - 20
        if remaining > 5:
            result = _try_explicit_rank16(n, m, p)
            if result is not None:
                return result
    
    # Try Rank 17
    elapsed = time.time() - START_TIME
    t17_budget = min(20, deadline - time.time() - 15)
    if t17_budget > 5:
        result = _search_rank_aggressive(n, m, p, 17, timeout=t17_budget, n_trials=80)
        if result is not None:
            return result
    
    # Fallback Rank 18
    return _construct_rank18(n, m, p)
