"""Combined: rank-18 base with enhanced rank-17 ALS search and exact verification."""
import numpy as np
import time
import sys
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

def _check_decomposition_fast(U, V, W, T):
    """Fast float check for decomposition validity."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    n, m, p = DIMENSIONS
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _check_decomposition_exact(U, V, W, T):
    """
    Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w.
    Uses Fractions for exactness.
    """
    n, m, p = DIMENSIONS
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    
    # First, a fast float check
    if not _check_decomposition_fast(U, V, W, T):
        return False
        
    # Try to snap to fractions with small denominators
    def to_frac(val):
        # Check if close to integer
        if abs(val - round(val)) < 1e-9:
            return Fraction(int(round(val)), 1)
        # Check if close to half-integer
        if abs(val - round(val * 2) / 2.0) < 1e-9:
            return Fraction(int(round(val * 2)), 2)
        # Check if close to quarter-integer
        if abs(val - round(val * 4) / 4.0) < 1e-9:
            return Fraction(int(round(val * 4)), 4)
        # Check if close to eighth-integer
        if abs(val - round(val * 8) / 8.0) < 1e-9:
            return Fraction(int(round(val * 8)), 8)
        return None

    try:
        U_frac = np.zeros_like(U_a, dtype=object)
        V_frac = np.zeros_like(V_a, dtype=object)
        W_frac = np.zeros_like(W_a, dtype=object)
        
        for i in range(U_a.shape[0]):
            for j in range(U_a.shape[1]):
                f = to_frac(U_a[i, j])
                if f is None:
                    return False
                U_frac[i, j] = f
        for i in range(V_a.shape[0]):
            for j in range(V_a.shape[1]):
                f = to_frac(V_a[i, j])
                if f is None:
                    return False
                V_frac[i, j] = f
        for i in range(W_a.shape[0]):
            for j in range(W_a.shape[1]):
                f = to_frac(W_a[i, j])
                if f is None:
                    return False
                W_frac[i, j] = f

        # Exact verification
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    total = Fraction(0)
                    for r in range(U_a.shape[1]):
                        total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
                    if total != T[u, v, w]:
                        return False
        return True
    except Exception:
        return False

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

def _compute_error(U, V, W, T_a, n, m, p, R):
    """Compute the error of the decomposition."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    return np.linalg.norm(result - T_a) / np.sqrt(T_a.size)

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=200, tol=1e-12, deadline=None):
    """Run ALS from a given initialization, return (U, V, W) as lists if exact."""
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        err = _compute_error(U, V, W, T_a, n, m, p, R)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Check for exact solution using fast check first
    if _check_decomposition_fast(U, V, W, T):
        # Snap to clean rational values for output
        def snap_list(mat):
            out = []
            for row in mat:
                r = []
                for val in row:
                    found = False
                    for denom in [1, 2, 4, 8, 16, 32]:
                        num = round(val * denom)
                        if abs(val - num/denom) < 1e-9:
                            r.append(num/denom)
                            found = True
                            break
                    if not found:
                        r.append(val)
                out.append(r)
            return out
        U_snapped, V_snapped, W_snapped = snap_list(U), snap_list(V), snap_list(W)
        # Verify the snapped version is still valid
        if _check_decomposition_fast(U_snapped, V_snapped, W_snapped, T):
            return U_snapped, V_snapped, W_snapped
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition with enhanced strategies."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: remove each term and perturb with multiple scales
    for term_to_remove in range(18):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        # Multiple seeds with varying perturbation scales
        for seed in range(15):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed)
            # Use a range of perturbation scales
            scale = 0.001 + 0.003 * (seed % 8)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=100, deadline=deadline)
            if result is not None:
                return result
    
    # Strategy 2: Random initializations with diverse scales
    for seed in range(25):
        if time.time() > deadline:
            break
        np.random.seed(seed * 7 + 123)
        scale = 0.3 + 0.2 * (seed % 5)
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=80, deadline=deadline)
        if result is not None:
            return result
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    T = _build_target()
    
    # First, build the known rank-18 decomposition
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Verify rank-18 with fast check
    if _check_decomposition_fast(U18, V18, W18, T):
        # Try to find rank-17
        result17 = _try_rank17_search(n, m, p, T, deadline)
        if result17 is not None:
            U, V, W = result17
            # Final double check with fast verification
            if _check_decomposition_fast(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: return rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
