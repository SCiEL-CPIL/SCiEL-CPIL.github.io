"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_eqs = n * m * p  # 42 equations
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*p, m*p, n*m) but more tightly, 
    # the rank is at least max(n*p, m*p, n*m) = max(14, 21, 6) = 21
    # But actually for matrix mult, rank >= max(n*p, m*p, n*m) is not quite right
    # A better lower bound: rank >= n*p (since output has n*p entries)
    # And rank >= m*p, rank >= n*m
    # So lower bound = max(n*p, m*p, n*m) = 21
    
    best_result = None
    
    # Start from a reasonable rank and go up
    for rank in range(30, n*m*p + 1):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of this rank
        # Use a random initialization and solve the nonlinear system
        # The equations are: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # This is a system of 42 equations with 41*rank unknowns
        # We can use least squares or try to solve iteratively
        
        # Strategy: fix W, solve for U and V linearly, or use alternating minimization
        # Actually, for exact decomposition, we need the tensor to be exactly represented
        
        # Let's try: set W to identity-like, then solve for U, V
        # Or use the fact that we can absorb scalings
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + rank)
            
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Alternating minimization: fix two, solve for third
            # The equation is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is trilinear, not bilinear, so alternating minimization is tricky
            
            # Alternative: use the tensor structure more directly
            # T[i,j,k] = e_{i,k}, so we need:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Let's define: for each c, the contribution is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
            # and we need the sum to equal T
            
            # This is equivalent to: the tensor T must be in the span of {U_c ⊗ V_c ⊗ W_c}
            
            # Use a different approach: since the tensor is sparse/structured,
            # let's try to find a rank-r decomposition by solving a linear system
            # after fixing one factor
            
            # Fix W, then the equation becomes:
            # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            # Let M_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c], then sum_c M_c[i,j,k] * W[k*n+i,c] = 1
            # This is linear in M_c and W_c but bilinear in U,V
            
            # Simpler: use the fact that for each c, we can define
            # A_c[i,j] = U[i*m+j,c], B_c[j,k] = V[j*p+k,c], and W_c[k,i] = W[k*n+i,c]
            # Then sum_c A_c[i,j] * B_c[j,k] * W_c[k,i] = 1
            
            # This is the "trace" form. The standard decomposition has
            # A_c = e_i e_j^T, B_c = e_j e_k^T, W_c = e_k e_i^T for each (i,j,k)
            
            # Let me try a direct numerical approach with scipy
            try:
                from scipy.optimize import least_squares
            except ImportError:
                break
            
            # Flatten the problem
            def residuals(x):
                U_flat = x[:nm*rank].reshape(nm, rank)
                V_flat = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_flat = x[nm*rank+mp*rank:].reshape(pn, rank)
                
                res = np.zeros(total_eqs)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            # Initial guess
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000, xtol=1e-10, ftol=1e-10)
                if result.success or np.max(np.abs(result.fun)) < 1e-8:
                    U_sol = result.x[:nm*rank].reshape(nm, rank)
                    V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Verify
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(rank):
                                    val += U_sol[i*m+j, c] * V_sol[j*p+k, c] * W_sol[k*n+i, c]
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-8:
                        best_result = {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
                        break
            except Exception:
                pass
        
        if best_result is not None:
            break
    
    # If no better found, return schoolbook
    if best_result is None:
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
    
    return best_result
# EVOLVE-BLOCK-END
