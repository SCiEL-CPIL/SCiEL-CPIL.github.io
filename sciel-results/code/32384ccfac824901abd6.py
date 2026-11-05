import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    timeout = 110
    n, m, p = 2, 3, 3
    
    def elapsed():
        return time.time() - start_time
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    def verify(U, V, W, T, tol=1e-9):
        Re = T - np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(Re)) < tol
    
    def build_rank17():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val
        
        set_U(0, 0, 0, 1); set_V(0, 0, 0, 1); set_W(0, 0, 0, 1)
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)
        
        # Cross terms
        set_U(0, 0, 7, 1); set_V(0, 2, 7, 1); set_W(2, 0, 7, 1)
        set_U(0, 1, 8, 1); set_V(1, 2, 8, 1); set_W(2, 0, 8, 1)
        set_U(1, 0, 9, 1); set_V(0, 2, 9, 1); set_W(2, 1, 9, 1)
        set_U(1, 1, 10, 1); set_V(1, 2, 10, 1); set_W(2, 1, 10, 1)
        set_U(0, 2, 11, 1); set_V(2, 0, 11, 1); set_W(0, 0, 11, 1)
        set_U(0, 2, 12, 1); set_V(2, 1, 12, 1); set_W(1, 0, 12, 1)
        set_U(0, 2, 13, 1); set_V(2, 2, 13, 1); set_W(2, 0, 13, 1)
        set_U(1, 2, 14, 1); set_V(2, 0, 14, 1); set_W(0, 1, 14, 1)
        set_U(1, 2, 15, 1); set_V(2, 1, 15, 1); set_W(1, 1, 15, 1)
        set_U(1, 2, 16, 1); set_V(2, 2, 16, 1); set_W(2, 1, 16, 1)
        
        return U, V, W
    
    def try_rank16():
        """Try to find a rank-16 decomposition by modifying the rank-17 one."""
        # The idea: try to combine two of the cross terms with the Strassen terms
        # or find a different combination that saves one rank
        
        # Let's try a direct approach: fix the first 7 terms (Strassen for 2x2x2)
        # and try to find 9 terms for the remaining 10 cross-term contributions
        
        # Actually, let's try a smarter approach: use the fact that the tensor has
        # specific structure and try random sign patterns for rank 16
        
        T = build_tensor()
        T_flat = T.flatten()
        
        # For rank r, we need U (6x r), V (9x r), W (6x r)
        # such that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        
        # The constraint is: for each (a,b,c) where T[a,b,c]=1,
        # sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # For each (a,b,c) where T[a,b,c]=0,
        # sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # This is a system of polynomial equations. Let's try to solve it
        # by fixing some entries and solving linearly for others.
        
        # Strategy: Fix U and V to have sparse ±1 entries, then solve for W linearly
        
        # For each term r, U[:,r] and V[:,r] are sparse vectors with ±1 entries
        # Let's enumerate possible sparse patterns
        
        # For a 6-element vector with exactly k nonzero entries (each ±1),
        # there are C(6,k) * 2^k patterns
        
        # For rank 16, let's try:
        # - 7 terms from Strassen (fixed)
        # - 9 terms to be determined
        
        # Actually, let's try a different approach: use the existing rank-17
        # and try to merge two terms into one by finding a linear combination
        
        U17, V17, W17 = build_rank17()
        
        # Try to see if any two terms can be merged
        # Term r and s can be merged if there exist coefficients alpha, beta
        # such that alpha*U[:,r] + beta*U[:,s] has the same support pattern
        # and similarly for V and W, and alpha*1 + beta*1 = 1 (for the target)
        
        # This is getting complex. Let's try a numerical optimization approach
        # with a smooth surrogate
        
        # Alternative: try to find rank 16 by solving a least-squares problem
        # with sparsity constraints
        
        # Let's try: fix the structure of U and V (sparse ±1), solve for W
        # The system is: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # For fixed U and V, this is linear in W!
        
        # So: for each (a,b,c), sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
        # This is a linear system in W[c,:] for each c
        
        # Let's try random sparse U and V, then solve for W
        
        np.random.seed(42)
        
        def try_sparse_decomposition(rank, max_nonzero_U=2, max_nonzero_V=2, max_nonzero_W=1):
            """Try to find a decomposition with sparse ±1 entries."""
            for attempt in range(500):
                if elapsed() > timeout - 5:
                    return None
                
                U = np.zeros((6, rank))
                V = np.zeros((9, rank))
                
                for r in range(rank):
                    # Random sparse pattern for U[:, r]
                    n_nonzero_U = np.random.randint(1, max_nonzero_U + 1)
                    idx_U = np.random.choice(6, n_nonzero_U, replace=False)
                    for idx in idx_U:
                        U[idx, r] = np.random.choice([-1, 1])
                    
                    # Random sparse pattern for V[:, r]
                    n_nonzero_V = np.random.randint(1, max_nonzero_V + 1)
                    idx_V = np.random.choice(9, n_nonzero_V, replace=False)
                    for idx in idx_V:
                        V[idx, r] = np.random.choice([-1, 1])
                
                # Now solve for W: for each c (0..5), we have a linear system
                # sum_r (U[:,r] * V[:,r])[a,b] * W[c,r] = T[a,b,c]
                # This gives 6*9=54 equations for 6 unknowns (W[c,0..rank-1])
                # Wait, that's over-determined. Let me reconsider.
                
                # Actually, for each c, W[c,:] is a vector of length rank.
                # The constraints are: for all (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # This is 54 equations for rank unknowns.
                
                # Build the linear system for each c
                W = np.zeros((6, rank))
                valid = True
                
                for c in range(6):
                    # Build matrix A_c where A_c[a*b + b, r] = U[a,r] * V[b,r]
                    A_c = np.zeros((54, rank))
                    b_c = np.zeros(54)
                    
                    for a in range(6):
                        for b in range(9):
                            idx = a * 9 + b
                            A_c[idx, :] = U[a, :] * V[b, :]
                            # Find which (i,j,k) this corresponds to
                            # a = i*m + j = 3*i + j
                            # b = j*p + k = 3*j + k
                            # c = k*n + i = 2*k + i
                            i = a // 3
                            j = a % 3
                            j_from_b = b // 3
                            k = b % 3
                            
                            if j == j_from_b:
                                c_expected = 2 * k + i
                                if c_expected == c:
                                    b_c[idx] = 1.0
                                else:
                                    b_c[idx] = 0.0
                            else:
                                b_c[idx] = 0.0
                    
                    # Solve A_c @ W[c,:] = b_c
                    # This is over-determined, use least squares
                    try:
                        W_c, residuals, rank_c, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
                        W[c, :] = W_c
                        
                        # Check if the solution is close to integer
                        if np.max(np.abs(W_c - np.round(W_c))) > 1e-6:
                            valid = False
                            break
                        
                        # Check exactness
                        recon = A_c @ W_c
                        if np.max(np.abs(recon - b_c)) > 1e-9:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Verify the full decomposition
                U_rounded = np.round(U)
                V_rounded = np.round(V)
                W_rounded = np.round(W)
                
                if verify(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded, V_rounded, W_rounded
            
            return None
        
        result = try_sparse_decomposition(16)
        if result is not None:
            return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
        
        return None
    
    T = build_tensor()
    
    # Try rank 16 first
    if elapsed() < timeout - 10:
        result16 = try_rank16()
        if result16 is not None:
            return result16
    
    # Fall back to rank 17
    U, V, W = build_rank17()
    
    if verify(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Naive rank 18
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
