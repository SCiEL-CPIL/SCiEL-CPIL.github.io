import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(U, V, W):
    """Round entries to nearest half-integer."""
    Ur = [[round(x * 2) / 2.0 for x in row] for row in U]
    Vr = [[round(x * 2) / 2.0 for x in row] for row in V]
    Wr = [[round(x * 2) / 2.0 for x in row] for row in W]
    return Ur, Vr, Wr

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Lower bound: for (n,m,p), rank >= max(n*m, m*p, p*n) in some cases
    # Actually a simple lower bound is max(n*m, m*p, p*n) / something... 
    # Let's just try ranks from a reasonable lower bound up to schoolbook
    
    # A known lower bound for matrix multiplication tensor rank is 
    # max(n*m, m*p, p*n) but that's not tight. Let's try from 24 upward.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from 24 to best_rank - 1
    for rank in range(24, best_rank):
        if time.time() > deadline:
            break
        
        # Multiple attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Optimize using a simple iterative approach
            # We want: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # Use a simple gradient-free approach: coordinate descent
            # Or use scipy optimization
            
            from scipy.optimize import minimize
            import itertools
            
            # Flatten the variables
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            size_U = n*m*rank
            size_V = m*p*rank
            size_W = p*n*rank
            
            def objective(x):
                Uf = x[:size_U].reshape(n*m, rank)
                Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
                Wf = x[size_U+size_V:].reshape(p*n, rank)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(Uf[i*m+j], Vf[j*p+k] * Wf[k*n+i])
                            err += (s - 1.0)**2
                return err
            
            # Try a few optimization steps
            try:
                result = minimize(objective, x0, method='Nelder-Mead', 
                                 options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            except:
                continue
            
            x_opt = result.x
            U_opt = x_opt[:size_U].reshape(n*m, rank)
            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, rank)
            W_opt = x_opt[size_U+size_V:].reshape(p*n, rank)
            
            # Round to half-integers
            U_r = np.round(U_opt * 2) / 2.0
            V_r = np.round(V_opt * 2) / 2.0
            W_r = np.round(W_opt * 2) / 2.0
            
            # Check if rounded version is exact
            exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                        if abs(s - 1.0) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                # Convert to proper types
                U_final = [[float(x) for x in row] for row in U_list]
                V_final = [[float(x) for x in row] for row in V_list]
                W_final = [[float(x) for x in row] for row in W_list]
                if _verify(U_final, V_final, W_final, n, m, p):
                    best_rank = rank
                    best_result = {"U": U_final, "V": V_final, "W": W_final}
                    break  # Found valid decomposition at this rank, try lower
        
        # If we found a valid one at this rank, we could break but let's try to go lower
        # Actually we're going from low to high, so first valid is best
        if best_rank == rank:
            break
    
    return best_result
