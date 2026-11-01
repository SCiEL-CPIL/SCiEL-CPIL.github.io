"""Rewritten program for matrix multiplication tensor decomposition 2x3x5."""
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Number of valid (i, j, k) triples = n*m*p = 30
    # We want rank r such that for all (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Use ansatz: try rank values from a smart lower bound up, and use optimization.
    # For 2x3x5, try rank 15, 18, 20, ...
    
    # We'll use a structured ansatz to make the problem more tractable.
    # 
    # Idea: For each pair (i, j), the k-dimension contributions must sum correctly.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Let's try a decomposition where we split k into pairs.
    # For k = 0,1,2,3,4, group them as (0,1), (2,3), (4).
    # For each group, use 2 terms (or 1 for singleton).
    # Total per (i,j): 2+2+1 = 5, same as schoolbook.
    # 
    # Better idea: use the identity for products.
    # For 2x2 case: (a0,a1) x (b0,b1) needs 7 terms (Strassen).
    # 
    # Let me just use scipy to find a decomposition of rank 20.
    #
    # Actually, let me try a different approach: use the fact that we can
    # share factors between different (i,j) pairs.
    #
    # Set W[k*n+i, r] = delta_{r, k*n+i} for r < p*n = 10 (W is identity-like)
    # Then we need: sum_r U[i*m+j, r] * V[j*p+k, r] * delta_{r, k*n+i} = 1
    # => For r = k*n+i: U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all i,j,k
    # But V[j*p+k, k*n+i] depends on j,p,k and k*n+i, which for fixed k,n,i is V[j*p+k, fixed]
    # This forces U and V to be specific, and rank would be at least p*n = 10, but we'd need
    # U[i*m+j, k*n+i] = 1/V[j*p+k, k*n+i], and for this to work for all j, 
    # V[j*p+k, k*n+i] must be independent of j (since U doesn't depend on j in a compatible way)... 
    # Actually U[i*m+j, k*n+i] depends on j via the row index i*m+j.
    # 
    # This approach gets complex. Let me just brute-force with a random initialization 
    # and Nelder-Mead or similar to minimize the residual.
    
    # Try rank = 15 first (a reasonable guess for improvement over 30)
    for rank in [15, 18, 20, 25, 30]:
        np.random.seed(42 + rank)
        
        # Flatten all parameters
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total = size_U + size_V + size_W
        
        x0 = np.random.randn(total) * 0.5
        x0 = np.where(np.abs(x0) < 0.1, 0, x0)  # sparse initialization
        
        def unpack(x):
            U = x[0:size_U].reshape(n*m, rank)
            V = x[size_U:size_U+size_V].reshape(m*p, rank)
            W = x[size_U+size_V:total].reshape(p*n, rank)
            return U, V, W
        
        def residual(x):
            U, V, W = unpack(x)
            errs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        errs.append(val - 1.0)
            return np.array(errs)
        
        def objective(x):
            r = residual(x)
            return np.sum(r * r)
        
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        U, V, W = unpack(res.x)
        max_err = max(abs(float(res.fun) ** 0.5), 1e-9)
        
        # Check if we got a good solution
        if objective(res.x) < 1e-12:
            # Convert to list of lists with exact values (round small values to 0)
            U_list = [[round(float(U[i][j]), 12) for j in range(rank)] for i in range(n*m)]
            V_list = [[round(float(V[i][j]), 12) for j in range(rank)] for i in range(m*p)]
            W_list = [[round(float(W[i][j]), 12) for j in range(rank)] for i in range(p*n)]
            
            # Clean up: set values close to 0 to 0, round to nearest 0.5
            def clean(mat):
                out = []
                for row in mat:
                    newrow = []
                    for v in row:
                        if abs(v) < 1e-9:
                            newrow.append(0)
                        else:
                            # Round to nearest 0.5
                            rounded = round(v * 2) / 2
                            if abs(rounded - v) < 1e-8:
                                newrow.append(rounded)
                            else:
                                newrow.append(v)
                    out.append(newrow)
                return out
            
            U_clean = clean(U_list)
            V_clean = clean(V_list)
            W_clean = clean(W_list)
            
            return {"U": U_clean, "V": V_clean, "W": W_clean}
    
    # Fallback: schoolbook
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
