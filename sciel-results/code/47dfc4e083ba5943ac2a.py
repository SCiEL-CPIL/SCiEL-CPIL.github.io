import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n, m, p, R):
    """Verify the decomposition exactly using Fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    u_val = Fraction(str(U[idx_u, r]))
                    v_val = Fraction(str(V[idx_v, r]))
                    w_val = Fraction(str(W[idx_w, r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _tile_decomposition(U_sub, V_sub, W_sub, n, m, p, n_sub, m_sub, p_sub, R_sub):
    """
    Tile a decomposition of M_{n_sub, m_sub, p_sub} to M_{n, m, p}
    where n=n_sub, m=m_sub, and p is a multiple of p_sub.
    """
    repeats = p // p_sub
    R_full = R_sub * repeats
    
    U_full = [[0] * R_full for _ in range(n * m)]
    V_full = [[0] * R_full for _ in range(m * p)]
    W_full = [[0] * R_full for _ in range(p * n)]
    
    # Fill U: same pattern repeated for each tile
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            for rep in range(repeats):
                r_offset = rep * R_sub
                for r in range(R_sub):
                    U_full[idx_u][r_offset + r] = U_sub[i * m_sub + j][r]
    
    # Fill V: each k-block uses the corresponding sub-V
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            rep = k // p_sub
            k_sub = k % p_sub
            r_offset = rep * R_sub
            idx_v_sub = j * p_sub + k_sub
            for r in range(R_sub):
                V_full[idx_v][r_offset + r] = V_sub[idx_v_sub][r]
    
    # Fill W: each k-block uses the corresponding sub-W
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            rep = k // p_sub
            k_sub = k % p_sub
            r_offset = rep * R_sub
            idx_w_sub = k_sub * n_sub + i
            for r in range(R_sub):
                W_full[idx_w][r_offset + r] = W_sub[idx_w_sub][r]
    
    return U_full, V_full, W_full

def _find_2x3x2_rank11(max_time=60):
    """
    Find a rank-11 decomposition for 2x3x2 using structured search.
    We fix U and V to be simple half-integer matrices and solve for W.
    """
    n, m, p = 2, 3, 2
    R = 11
    start = time.time()
    
    # Try various structured initializations
    # Strategy: fix U and V to be simple patterns, solve for W exactly
    
    # Try 1: U and V with entries in {-1, -0.5, 0, 0.5, 1}
    half_ints = [-1, -0.5, 0, 0.5, 1]
    
    # We'll try a few structured patterns
    patterns = []
    
    # Pattern 1: random half-integers
    np.random.seed(42)
    for seed in range(50):
        if time.time() - start > max_time:
            break
        np.random.seed(seed)
        U = np.array(np.random.choice(half_ints, size=(n*m, R)), dtype=float)
        V = np.array(np.random.choice(half_ints, size=(m*p, R)), dtype=float)
        
        # Check if we can solve for W
        # For each (i,k) pair, we need: sum_j U[i*m+j] * V[j*p+k] * W_vec = 1
        # This is a linear system in W_vec
        
        W = np.zeros((p*n, R))
        valid = True
        
        for i in range(n):
            for k in range(p):
                idx_w = k * n + i
                # Build the linear system: M @ W_vec = ones(m)
                # where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                M = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                
                # Solve M @ W_vec = ones(m)
                # Use least squares
                try:
                    W_vec, residuals, rank, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    # Check if the solution is close to half-integers
                    W_rounded = np.round(W_vec * 2) / 2
                    # Verify
                    if np.allclose(M @ W_rounded, np.ones(m), atol=1e-10):
                        W[idx_w, :] = W_rounded
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Final exact verification
            if _verify_exact(U, V, W, n, m, p, R):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _find_2x3x3_rank11(max_time=60):
    """
    Find a rank-11 decomposition for 2x3x3 using structured search.
    """
    n, m, p = 2, 3, 3
    R = 11
    start = time.time()
    
    half_ints = [-1, -0.5, 0, 0.5, 1]
    
    for seed in range(50):
        if time.time() - start > max_time:
            break
        np.random.seed(seed)
        U = np.array(np.random.choice(half_ints, size=(n*m, R)), dtype=float)
        V = np.array(np.random.choice(half_ints, size=(m*p, R)), dtype=float)
        
        W = np.zeros((p*n, R))
        valid = True
        
        for i in range(n):
            for k in range(p):
                idx_w = k * n + i
                M = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                
                try:
                    W_vec, residuals, rank, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_rounded = np.round(W_vec * 2) / 2
                    if np.allclose(M @ W_rounded, np.ones(m), atol=1e-10):
                        W[idx_w, :] = W_rounded
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            if _verify_exact(U, V, W, n, m, p, R):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strategy 1: Find rank-11 decomposition for 2x3x2, then tile to 2x3x6 (rank 33)
    # This is better than schoolbook (36) if we can find it
    
    result_232 = _find_2x3x2_rank11(max_time=60)
    
    if result_232 is not None:
        U_sub, V_sub, W_sub = result_232
        n_sub, m_sub, p_sub = 2, 3, 2
        R_sub = 11
        
        U_full, V_full, W_full = _tile_decomposition(
            U_sub, V_sub, W_sub, n, m, p, n_sub, m_sub, p_sub, R_sub)
        
        # Verify the full decomposition
        U_np = np.array(U_full, dtype=float)
        V_np = np.array(V_full, dtype=float)
        W_np = np.array(W_full, dtype=float)
        R_full = R_sub * (p // p_sub)
        
        if _verify_exact(U_np, V_np, W_np, n, m, p, R_full):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Strategy 2: Find rank-11 decomposition for 2x3x3, then tile to 2x3x6 (rank 22)
    if time.time() < deadline:
        result_233 = _find_2x3x3_rank11(max_time=40)
        
        if result_233 is not None:
            U_sub, V_sub, W_sub = result_233
            n_sub, m_sub, p_sub = 2, 3, 3
            R_sub = 11
            
            U_full, V_full, W_full = _tile_decomposition(
                U_sub, V_sub, W_sub, n, m, p, n_sub, m_sub, p_sub, R_sub)
            
            U_np = np.array(U_full, dtype=float)
            V_np = np.array(V_full, dtype=float)
            W_np = np.array(W_full, dtype=float)
            R_full = R_sub * (p // p_sub)
            
            if _verify_exact(U_np, V_np, W_np, n, m, p, R_full):
                return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
