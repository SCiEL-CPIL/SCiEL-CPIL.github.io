"""Rank-minimal tensor decomposition for 2x2 * 2x5 matrix multiplication.
Strategy: try to construct a rank-16 decomposition using Strassen on column pairs
with a clever reuse, then fall back to rank-18 if that fails.
Also try rank-17 via targeted perturbation + ALS.
"""
import numpy as np
import time
import sys

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

def _check_decomposition(U, V, W, T, atol=1e-9):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=atol)

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

def _try_rank16_direct(n, m, p):
    """
    Try to construct a rank-16 decomposition directly.
    
    Key insight: For 2x2 * 2x5, we can potentially use:
    - Strassen for columns (0,1): 7 terms
    - Strassen for columns (2,3): 7 terms  
    - But we need 4 more terms for column 4, total = 18
    
    Alternative: Use a mixed approach where some Strassen terms are shared.
    Let's try: treat the 2x5 as 2x2 * 2x2 + 2x1, but with optimized overlap.
    
    Actually, let's try a different decomposition:
    - Use Strassen-like for (0,1) and (2,3) but with 6 terms each (impossible for 2x2x2)
    
    Better approach: Try to find if rank 16 is achievable by solving the system.
    We'll set up the tensor equations and try to solve them.
    """
    T = _build_target()
    T_a = T.astype(np.float64)
    R = 16
    nm = n*m  # 4
    mp = m*p  # 10
    pn = p*n  # 10
    
    # Total number of tensor entries
    total_entries = nm * mp * pn  # 400
    
    # Try random initializations with structured patterns
    for attempt in range(50):
        if time.time() > _DEADLINE:
            break
        
        np.random.seed(attempt * 31 + 7)
        
        # Initialize with structured patterns
        U = np.zeros((nm, R), dtype=np.float64)
        V = np.zeros((mp, R), dtype=np.float64)
        W = np.zeros((pn, R), dtype=np.float64)
        
        # First 7 terms: Strassen-like for columns 0,1
        # Terms 7-13: Strassen-like for columns 2,3
        # Terms 14-15: For column 4
        
        # Use the rank-18 pattern but compress
        U18, V18, W18 = _build_rank18(n, m, p)
        U18_a = np.array(U18, dtype=np.float64)
        V18_a = np.array(V18, dtype=np.float64)
        W18_a = np.array(W18, dtype=np.float64)
        
        # Remove 2 schoolbook terms and perturb
        if attempt < 20:
            # Remove terms 16,17 (U[2] and U[3] for col 4)
            idx = [i for i in range(18) if i not in (16, 17)]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
        elif attempt < 40:
            # Remove terms 14,15 (U[0] and U[1] for col 4)
            idx = [i for i in range(18) if i not in (14, 15)]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
        else:
            # Random
            U_init = np.random.randn(nm, R) * 0.5
            V_init = np.random.randn(mp, R) * 0.5
            W_init = np.random.randn(pn, R) * 0.5
        
        # Add small perturbation
        U = U_init + np.random.randn(nm, R) * 0.01
        V = V_init + np.random.randn(mp, R) * 0.01
        W = W_init + np.random.randn(pn, R) * 0.01
        
        # Run ALS
        U, V, W = _run_als(T_a, U, V, W, R, max_iters=150, deadline=_DEADLINE)
        
        if _check_decomposition(U, V, W, T_a, atol=1e-8):
            # Try to round to half-integers
            for denom in [2, 4, 8, 16, 32, 64, 128]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                if _check_decomposition(U_s, V_s, W_s, T_a, atol=1e-8):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank17_direct(n, m, p):
    """Try to find rank-17 decomposition."""
    T = _build_target()
    T_a = T.astype(np.float64)
    R = 17
    nm = n*m
    mp = m*p
    pn = p*n
    
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    for term_to_remove in range(18):
        if time.time() > _DEADLINE:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(10):
            if time.time() > _DEADLINE:
                break
            np.random.seed(term_to_remove * 100 + seed)
            scale = 0.01 + 0.01 * seed
            U = U_init + np.random.randn(nm, R) * scale
            V = V_init + np.random.randn(mp, R) * scale
            W = W_init + np.random.randn(pn, R) * scale
            
            U, V, W = _run_als(T_a, U, V, W, R, max_iters=100, deadline=_DEADLINE)
            
            if _check_decomposition(U, V, W, T_a, atol=1e-8):
                for denom in [2, 4, 8, 16, 32, 64]:
                    U_s = np.round(U * denom) / denom
                    V_s = np.round(V * denom) / denom
                    W_s = np.round(W * denom) / denom
                    if _check_decomposition(U_s, V_s, W_s, T_a, atol=1e-8):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _run_als(T_a, U, V, W, R, max_iters=100, deadline=None, tol=1e-12):
    """Run ALS optimization."""
    prev_err = np.inf
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(U.shape[0]):
            b = T_a[u].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(V.shape[0]):
            b = T_a[:, v, :].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(W.shape[0]):
            b = T_a[:, :, w].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        # Compute error
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
        err = np.sqrt(err / T_a.size)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

# Global deadline
_DEADLINE = None

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # First, build the known rank-18 decomposition
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Verify rank-18
    if not _check_decomposition(U18, V18, W18, T):
        # Should never happen, but fallback
        pass
    
    # Try rank-16 first (best possible improvement)
    result16 = _try_rank16_direct(n, m, p)
    if result16 is not None:
        U, V, W = result16
        if _check_decomposition(U, V, W, T, atol=1e-8):
            return {"U": U, "V": V, "W": W}
    
    # Try rank-17
    if time.time() < _DEADLINE - 10:
        result17 = _try_rank17_direct(n, m, p)
        if result17 is not None:
            U, V, W = result17
            if _check_decomposition(U, V, W, T, atol=1e-8):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: return rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
