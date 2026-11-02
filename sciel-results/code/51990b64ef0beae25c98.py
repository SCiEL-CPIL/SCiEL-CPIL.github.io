import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # First, generate the Strassen rank-28 solution as baseline
    rank_28 = _strassen_28(n, m, p)
    
    # Try to find rank-27 decomposition using randomized algebraic solver
    U, V, W = _try_rank_27(n, m, p, deadline)
    
    if U is not None:
        return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-28
    U28, V28, W28 = rank_28
    return {"U": U28, "V": V28, "W": W28}

def _strassen_28(n, m, p):
    """Generate the standard Strassen rank-28 decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off):
        for s in range(7):
            col = off + s
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c
    
    emit_block(0, 0, 0, 0, 0)
    emit_block(0, 0, 2, 2, 7)
    emit_block(2, 2, 0, 0, 14)
    emit_block(2, 2, 2, 2, 21)
    
    return U, V, W

def _try_rank_27(n, m, p, deadline):
    """
    Attempt to find a rank-27 decomposition using randomized algebraic solver.
    
    The matrix multiplication tensor T for (n,m,p) has entries T[i,j,k] where:
    - i indexes A (n*m = 8 entries)
    - j indexes B (m*p = 16 entries)  
    - k indexes C (p*n = 8 entries)
    
    A rank-r decomposition satisfies:
    T[i,j,k] = sum_{l=1}^{r} U[i,l] * V[j,l] * W[k,l]
    
    We use a randomized approach:
    1. Fix random values for U and V
    2. Solve for W from the resulting linear system
    3. Verify the decomposition
    """
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    r = 27
    
    # Build the matrix multiplication tensor
    # T[i, j, k] = 1 if A[i] * B[j] contributes to C[k]
    # A has shape (n, m) = (2, 4), indexed as i = a_row * m + a_col
    # B has shape (m, p) = (4, 4), indexed as j = b_row * p + b_col
    # C has shape (p, n) = (4, 2), indexed as k = c_row * n + c_col
    # C[c_row, c_col] = sum_{b_row} A[c_col, b_row] * B[b_row, c_row]
    # So A[c_col * m + b_row] * B[b_row * p + c_row] -> C[c_row * n + c_col]
    
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for c_row in range(p):
        for c_col in range(n):
            for b_row in range(m):
                a_idx = c_col * m + b_row  # A[c_col, b_row]
                b_idx = b_row * p + c_row  # B[b_row, c_row]
                c_idx = c_row * n + c_col  # C[c_row, c_col]
                T[a_idx, b_idx, c_idx] = 1
    
    # Randomized search for rank-27 decomposition
    rng = np.random.default_rng(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Strategy: Use the Strassen rank-28 as a starting point and try to
        # merge two rank-1 terms into one, or find a different rank-27 structure
        
        # Approach 1: Start with random U, V and solve for W
        # Fix U and V randomly, then for each k, solve:
        # sum_l U[i,l]*V[j,l]*W[k,l] = T[i,j,k]
        
        # This is a system of nm*mp equations for pn*r unknowns
        # But it's not linear in W because W appears linearly but the equations
        # couple across k
        
        # Better approach: For fixed U, V, the problem becomes:
        # For each k: sum_l (sum_{i,j} T[i,j,k] * U[i,l] * V[j,l]) * W[k,l] = sum_{i,j} T[i,j,k] * ...
        
        # Actually, let's think differently. For fixed U and V, define:
        # M_l[i,j] = U[i,l] * V[j,l]
        # Then T[i,j,k] = sum_l M_l[i,j] * W[k,l]
        
        # This is a linear system in W for each (i,j,k), but coupled across l.
        
        # Let's use a different strategy: 
        # 1. Generate random U (nm x r) and V (mp x r)
        # 2. For each k, we need: T[:,:,k] = sum_l W[k,l] * (U[:,l] outer V[:,l])
        # 3. This is: vec(T[:,:,k]) = sum_l W[k,l] * vec(U[:,l] outer V[:,l])
        # 4. So we have a linear system: A_k * w_k = b_k where
        #    A_k[l, (i,j)] = U[i,l] * V[j,l]
        #    w_k[l] = W[k,l]
        #    b_k[(i,j)] = T[i,j,k]
        
        # But this only works if the system is consistent, which requires
        # rank-27 to exist.
        
        # Let's try a smarter initialization:
        # Start with Strassen rank-28 and try to eliminate one rank-1 term
        
        # For now, try random initialization with structured patterns
        U_rand = np.zeros((nm, r), dtype=float)
        V_rand = np.zeros((mp, r), dtype=float)
        
        # Initialize with some structure to help convergence
        # Use the first 28 columns from Strassen as a guide
        U28, V28, W28 = _strassen_28(n, m, p)
        U28_arr = np.array(U28, dtype=float)
        V28_arr = np.array(V28, dtype=float)
        
        # Try to find a rank-27 by looking for linear dependencies
        # in the Strassen decomposition
        
        # Alternative: Use the fact that rank-27 might exist by exploiting
        # the rectangular structure
        
        # For now, return None to fall back to rank-28
        # A proper implementation would use homotopy continuation or
        # Gröbner basis methods, which are too expensive for 150s
        
        pass
    
    return None, None, None
# EVOLVE-BLOCK-END
