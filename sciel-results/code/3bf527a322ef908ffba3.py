"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # For multiplying an n x m matrix by an m x p matrix,
    # the output is n x p.
    # The tensor T[i, j, k, l] where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # represents (A * B)[i, l] = sum_j A[i, j] * B[j, l]
    # So T[i, j, k, l] = delta_{j, k}
    
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = delta_{j, k}
    
    # Let's think about this more carefully.
    # U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    
    # The constraint is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + l, r] = delta_{j,k} for all i,j,k,l
    
    # Note that the left side must be independent of i and l for it to equal delta_{j,k}.
    # This means: for fixed j,k, the sum must not depend on i or l.
    
    # One approach: make U depend only on j (not i), V depend only on j and k, 
    # and W depend only on k (not l). Then:
    # U[i*m + j, r] = u[j, r]
    # V[j*p + k, r] = v[j, k, r]  
    # W[k*n + l, r] = w[k, r]
    
    # Then sum_r u[j,r] * v[j,k,r] * w[k,r] = delta_{j,k}
    
    # This is like a rank-R decomposition of the identity matrix in some sense.
    
    # Simplest: R = m = 2
    # Let u[j, r] = delta_{j,r}, v[j,k,r] = delta_{j,r} * delta_{k,r}, w[k,r] = delta_{k,r}
    # Then sum_r delta_{j,r} * delta_{j,r} * delta_{k,r} * delta_{k,r} = sum_r delta_{j,r} * delta_{k,r} = delta_{j,k}
    # Wait, that's not quite right. Let me redo.
    
    # sum_r u[j,r] * v[j,k,r] * w[k,r]
    # = sum_r delta_{j,r} * (delta_{j,r} * delta_{k,r}) * delta_{k,r}
    # = sum_r delta_{j,r} * delta_{j,r} * delta_{k,r} * delta_{k,r}
    # = sum_r delta_{j,r} * delta_{k,r}  (since delta^2 = delta)
    # = delta_{j,k}
    
    # Yes! That works. So R = m = 2.
    
    R = m  # rank = 2
    
    # Build U: shape (n*m, R)
    # U[i*m + j, r] = delta_{j, r}
    U = [[0] * R for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for r in range(R):
                U[i * m + j][r] = 1 if j == r else 0
    
    # Build V: shape (m*p, R)
    # V[j*p + k, r] = delta_{j, r} * delta_{k, r}
    V = [[0] * R for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            for r in range(R):
                V[j * p + k][r] = 1 if (j == r and k == r) else 0
    
    # Build W: shape (p*n, R)
    # W[k*n + l, r] = delta_{k, r}
    W = [[0] * R for _ in range(p * n)]
    for k in range(p):
        for l in range(n):
            for r in range(R):
                W[k * n + l][r] = 1 if k == r else 0
    
    # Verify the solution
    def verify(U, V, W):
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + l][r]
                        T[i, j, k, l] = val
        
        T_target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_target[i, j, k, l] = 1.0 if j == k else 0.0
        
        return np.max(np.abs(T - T_target))
    
    err = verify(U, V, W)
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
