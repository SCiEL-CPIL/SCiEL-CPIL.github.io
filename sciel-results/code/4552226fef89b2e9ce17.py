import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    total = n*m*p  # 72
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i, j, k in indices:
            s = 0.0
            for rr in range(r):
                s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    # Try to find a decomposition with rank < 72
    # Start from rank 72 and try to reduce
    
    # First, let's try a numerical approach for specific ranks
    import random
    
    best_rank = total
    best_result = None
    
    # Try ranks from a reasonable lower bound
    # The rank must be at least max(nm, mp, pn) = 24, but likely higher
    for r in range(40, total + 1):
        found = False
        for trial in range(3):
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Use gradient descent or least squares
            # For speed, use a simple iterative approach
            
            # Actually, let's just try to see if rounding the random init works
            # (unlikely, but cheap to check)
            
            # Better: use the fact that we can solve for W given U and V
            # For each (i,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # This is a linear system in W for fixed U, V
            
            # For fixed U, V, the equations are:
            # For each (i, k): sum_j ... no wait, it's for each (i,j,k)
            
            # Let me reorganize. For fixed U, V, we have equations:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            # This is linear in W!
            # For each (i,k), we have m equations (one for each j):
            # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # But W[k*n+i, r] is the same for all j, so we have:
            # For each (i,k): the vector A_{i,k} where A_{i,k}[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # And we need: A_{i,k} @ W[:, k*n+i] = 1 (vector of ones of length m)
            
            # So for each (i,k), we solve: A_{i,k} @ w_{i,k} = 1_m
            # where w_{i,k} = W[:, k*n+i]
            
            # This is a linear system of m equations in r unknowns.
            # If r >= m, it's underdetermined and has solutions.
            
            # Let's solve this!
            W_sol = np.zeros((pn, r))
            solvable = True
            for i in range(n):
                for k in range(p):
                    # Build A: m x r matrix
                    A = np.zeros((m, r))
                    for j in range(m):
                        for rr in range(r):
                            A[j, rr] = U[i*m+j, rr] * V[j*p+k, rr]
                    # Solve A @ w = ones(m)
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_sol[k*n+i, :] = w
                    except:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if solvable:
                # Verify
                max_err = 0.0
                for i, j, k in indices:
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W_sol[k*n+i, rr]
                    max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-8:
                    # Round to half-integers
                    U_h = np.round(U * 2) / 2
                    V_h = np.round(V * 2) / 2
                    W_h = np.round(W_sol * 2) / 2
                    
                    # Verify after rounding
                    max_err2 = 0.0
                    for i, j, k in indices:
                        s = 0.0
                        for rr in range(r):
                            s += U_h[i*m+j, rr] * V_h[j*p+k, rr] * W_h[k*n+i, rr]
                        max_err2 = max(max_err2, abs(s - 1.0))
                    
                    if max_err2 < 1e-9:
                        best_rank = r
                        best_result = (U_h.tolist(), V_h.tolist(), W_h.tolist())
                        found = True
                        break
        
        if found:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = total
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
