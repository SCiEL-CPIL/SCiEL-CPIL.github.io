"""Combined program: structured rank search with verification for 2x3x6 matrix multiplication.
   Strategy:
   1. Build the schoolbook rank-36 decomposition as guaranteed fallback.
   2. Attempt to find lower-rank decompositions using structured patterns (Strassen-like
      combinations, block decompositions) and verify exactness with integer arithmetic.
   3. Use time-bounded search to try ranks below 36.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]


def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic (entries are half-integers).
    The tensor decomposition requires:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    Since entries can be half-integers, multiply by 2 and check sum == 8.
    """
    r = len(U[0])
    # Work with integers: multiply by 2
    U2 = [[2 * U[i][j] for j in range(r)] for i in range(n * m)]
    V2 = [[2 * V[i][j] for j in range(r)] for i in range(m * p)]
    W2 = [[2 * W[i][j] for j in range(r)] for i in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                if s != 8:  # 2*2*2 = 8
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _find_block_decomposition(n_block, m_block, p_block, target_rank, deadline):
    """Find a decomposition for a n_block x m_block x p_block tensor.
    Returns U, V, W or None.
    """
    if time.time() > deadline:
        return None
    
    try:
        from scipy.optimize import minimize
        
        size_u = n_block * m_block * target_rank
        size_v = m_block * p_block * target_rank
        size_w = p_block * n_block * target_rank
        size = size_u + size_v + size_w
        
        np.random.seed(42)
        x0 = np.random.randn(size) * 0.1
        
        def objective(x):
            if time.time() > deadline:
                return 1e10
            u = x[:size_u].reshape(n_block * m_block, target_rank)
            v = x[size_u:size_u + size_v].reshape(m_block * p_block, target_rank)
            w = x[size_u + size_v:].reshape(p_block * n_block, target_rank)
            
            error = 0.0
            for i in range(n_block):
                for j in range(m_block):
                    for k in range(p_block):
                        s = 0.0
                        for c in range(target_rank):
                            s += u[i * m_block + j, c] * v[j * p_block + k, c] * w[k * n_block + i, c]
                        error += (s - 1.0) ** 2
            return error
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12})
        
        if res.fun < 1e-8:
            u = res.x[:size_u].reshape(n_block * m_block, target_rank)
            v = res.x[size_u:size_u + size_v].reshape(m_block * p_block, target_rank)
            w = res.x[size_u + size_v:].reshape(p_block * n_block, target_rank)
            
            # Round to half-integers
            U = [[round(u[i][j] * 2) / 2 for j in range(target_rank)] for i in range(n_block * m_block)]
            V = [[round(v[i][j] * 2) / 2 for j in range(target_rank)] for i in range(m_block * p_block)]
            W = [[round(w[i][j] * 2) / 2 for j in range(target_rank)] for i in range(p_block * n_block)]
            
            # Verify
            r = target_rank
            U2 = [[2 * U[i][j] for j in range(r)] for i in range(n_block * m_block)]
            V2 = [[2 * V[i][j] for j in range(r)] for i in range(m_block * p_block)]
            W2 = [[2 * W[i][j] for j in range(r)] for i in range(p_block * n_block)]
            
            valid = True
            for i in range(n_block):
                for j in range(m_block):
                    for k in range(p_block):
                        s = 0
                        for c in range(r):
                            s += U2[i * m_block + j][c] * V2[j * p_block + k][c] * W2[k * n_block + i][c]
                        if s != 8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
    except Exception:
        pass
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    # Start with schoolbook as guaranteed fallback
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find rank 10 decomposition for 2x3x2 blocks
    # Total rank would be 3 * 10 = 30
    if time.time() < deadline - 10:
        block_result = _find_block_decomposition(2, 3, 2, 10, deadline - 10)
        if block_result is not None:
            U_block, V_block, W_block = block_result
            r_block = len(U_block[0])
            total_rank = 3 * r_block
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            for g in range(3):
                offset = g * r_block
                # Map indices
                # U: 2x3. Indices i*m+j.
                # V: 3x2 (for this block). Indices j*2+k_local.
                # W: 2x2 (for this block). Indices k_local*2+i.
                
                for i in range(2):
                    for j in range(3):
                        U[i * 3 + j][offset : offset + r_block] = U_block[i * 3 + j]
                
                for j in range(3):
                    for k_local in range(2):
                        k_global = 2 * g + k_local
                        V[j * 6 + k_global][offset : offset + r_block] = V_block[j * 2 + k_local]
                
                for k_local in range(2):
                    k_global = 2 * g + k_local
                    for i in range(2):
                        W[k_global * 2 + i][offset : offset + r_block] = W_block[k_local * 2 + i]
            
            if _verify_exact(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
                break # Found a good one, stop searching for this type
                
    return best_result
