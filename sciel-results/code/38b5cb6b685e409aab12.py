import numpy as np
from scipy.linalg import solve as sp_solve

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n  # 6, 18, 12
    
    # The tensor T is defined by T[i, j, k] = 1 if i < n, j < m, k < p
    # We want to find rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^n (flattened to R^{nm} with zeros outside), etc.
    
    # The bilinear equations are:
    # For each (i, j, k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # We use an alternating minimization approach (a numerical solver for the bilinear system).
    # This is a heuristic search for a valid decomposition.
    
    # Lower bound for rank is max(n*m, m*p, p*n) is not quite right, but rank >= max(n, m, p) is trivial.
    # A better lower bound is the border rank, but we don't know it.
    # Schoolbook rank is 36.
    
    # We will try to find a solution for r < 36.
    # If found, return it. Otherwise, return schoolbook.
    
    # For efficiency and to ensure we finish within 120s, we limit the search.
    # We'll try r from 35 down to 30 (a reasonable range).
    
    # Note: It is highly likely that the minimal rank is 36 for 2x3x6,
    # but we must attempt the search as per the directive's spirit.
    
    # The alternating minimization works as follows:
    # 1. Initialize U, V, W randomly.
    # 2. Fix V, W. Solve for U: For each (i,j), U[i,j,:] is the solution to
    #    sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c]) = 1
    #    This is a linear system: U[i,j,:] * A_{ij} = 1, where A_{ij} is a vector of length r.
    #    Actually, it's a scalar equation for each (i,j,k):
    #    sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    #    This is a system of P equations (one for each k) with r unknowns (U[i,j,:]).
    #    We can set up a linear system: B_{ij} U[i,j,:]^T = 1
    #    where B_{ij} is a P x r matrix with B_{ij}[k, c] = V[j,k,c] * W[k,i,c].
    #    We solve this in the least-squares sense.
    
    # Similarly for V and W.
    
    # Let's implement this.
    
    def try_rank(r, max_iters=1000, tol=1e-10, seed=42):
        np.random.seed(seed)
        U = np.random.randn(N, r)
        V = np.random.randn(M, r)
        W = np.random.randn(P, r)
        
        # Precompute indices
        # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        
        for it in range(max_iters):
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # B_{ij} is P x r
                    B = np.zeros((p, r))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve B @ U[i*m+j, :]^T = 1 (vector of ones of length p)
                    # This is a least squares problem.
                    try:
                        sol, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = sol
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # B_{jk} is N x r
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = sol
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # B_{ki} is M x r
                    B = np.zeros((m, r))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = sol
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Check residual
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < tol:
                return U, V, W, True
        
        return U, V, W, False
    
    # Try to find a solution for r < 36
    for r in range(35, 25, -1):
        U, V, W, success = try_rank(r)
        if success:
            # Verify the solution
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-8:
                # Round to half-integers if possible, otherwise keep as is
                # The problem allows integer/half-integer entries, but we can return floats
                # if they are close to such values.
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # If no solution found, return schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
