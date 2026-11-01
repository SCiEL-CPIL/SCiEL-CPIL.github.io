"""Rank-minimal tensor decomposition for 2x2x5 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

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

def _check_decomposition_exact(U, V, W, T):
    """Check exact decomposition using Fractions."""
    n, m, p = DIMENSIONS
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    R = U_a.shape[1]
    
    # Fast float check first
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    if not np.allclose(result, T.astype(np.float64), atol=1e-9):
        return False
    
    # Try to snap to simple rationals
    def to_frac(val):
        for denom in [1, 2, 4, 8]:
            num = round(val * denom)
            if abs(val - num/denom) < 1e-9:
                return Fraction(num, denom)
        return None
    
    try:
        U_frac = np.zeros((n*m, R), dtype=object)
        V_frac = np.zeros((m*p, R), dtype=object)
        W_frac = np.zeros((p*n, R), dtype=object)
        
        for i in range(n*m):
            for j in range(R):
                f = to_frac(U_a[i, j])
                if f is None:
                    return False
                U_frac[i, j] = f
        for i in range(m*p):
            for j in range(R):
                f = to_frac(V_a[i, j])
                if f is None:
                    return False
                V_frac[i, j] = f
        for i in range(p*n):
            for j in range(R):
                f = to_frac(W_a[i, j])
                if f is None:
                    return False
                W_frac[i, j] = f
        
        # Exact verification
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    total = Fraction(0)
                    for r in range(R):
                        total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
                    if total != T[u, v, w]:
                        return False
        return True
    except Exception:
        return np.allclose(result, T.astype(np.float64), atol=1e-12)

def _build_rank18(n, m, p):
    """Known rank-18 decomposition: Strassen for cols (0,1) and (2,3), schoolbook for col 4."""
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        # Standard Strassen for 2x2x2
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
    for i in range(2):
        for j in range(2):
            r = 14 + i*2 + j
            U[i*m+j][r] = 1
            V[j*p+4][r] = 1
            W[4*n+i][r] = 1
    
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
        sol, _, _, _ = np.linalg.lstsq(VvW_flat, b, rcond=None)
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
        sol, _, _, _ = np.linalg.lstsq(Uw_flat, b, rcond=None)
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
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=300, tol=1e-14):
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        result = np.zeros_like(T_a, dtype=np.float64)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.linalg.norm(result - T_a) / np.sqrt(T_a.size)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    if _check_decomposition_exact(U, V, W, T):
        def snap(mat):
            out = []
            for row in mat:
                r = []
                for val in row:
                    found = False
                    for denom in [1, 2, 4, 8, 16]:
                        num = round(val * denom)
                        if abs(val - num/denom) < 1e-9:
                            r.append(num/denom)
                            found = True
                            break
                    if not found:
                        r.append(val)
                out.append(r)
            return out
        return snap(U), snap(V), snap(W)
    return None

def _try_rank_search(n, m, p, T, target_rank, deadline, num_trials=50):
    """Try to find a decomposition of target_rank."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: Remove terms from rank-18 and perturb
    if target_rank == 17:
        for term_to_remove in range(18):
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
                scale = 0.0005 + 0.001 * (seed % 5)
                U_pert = U_init + np.random.randn(4, 17) * scale
                V_pert = V_init + np.random.randn(10, 17) * scale
                W_pert = W_init + np.random.randn(10, 17) * scale
                result = _als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=200)
                if result is not None:
                    return result
    
    # Strategy 2: Random initializations
    for seed in range(num_trials):
        if time.time() > deadline:
            break
        np.random.seed(seed * 13 + 7)
        U_init = np.random.randn(4, target_rank) * 0.5
        V_init = np.random.randn(10, target_rank) * 0.5
        W_init = np.random.randn(10, target_rank) * 0.5
        result = _als_from_init(U_init, V_init, W_init, T, n, m, p, target_rank, max_iters=150)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    T = _build_target()
    
    # Build known rank-18
    U18, V18, W18 = _build_rank18(n, m, p)
    best_result = {"U": U18, "V": V18, "W": W18}
    
    # Try rank 17
    if time.time() < deadline - 20:
        result17 = _try_rank_search(n, m, p, T, 17, deadline - 20)
        if result17 is not None:
            U, V, W = result17
            if _check_decomposition_exact(U, V, W, T):
                best_result = {"U": U, "V": V, "W": W}
    
    # Try rank 16
    if time.time() < deadline - 10:
        result16 = _try_rank_search(n, m, p, T, 16, deadline - 10, num_trials=30)
        if result16 is not None:
            U, V, W = result16
            if _check_decomposition_exact(U, V, W, T):
                best_result = {"U": U, "V": V, "W": W}
    
    return best_result
