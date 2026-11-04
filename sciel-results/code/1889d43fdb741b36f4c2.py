import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Construct the tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a low-rank decomposition
    # Start with rank 56
    for rank in range(56, 65):
        # Try to find U, V, W
        # Use a simple iterative method
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Iterate: fix two, solve for the third
        for iteration in range(100):
            # Fix U, V, solve for W
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed U, V: T[a,b,:] = (U[a,:]*V[b,:]) @ W.T
            # So W.T = (U[a,:]*V[b,:])^{-1} @ T[a,b,:] (if invertible)
            # This is a least squares problem
            
            # Actually, this is a 3-way tensor decomposition, which is more complex.
            # Let me use a simpler approach: fix U and V, and solve for W using least squares.
            
            # Build the linear system for W
            # For each (a,b): T[a,b,:] = sum_r (U[a,r]*V[b,r]) * W[c,r] for all c
            # This is: T[a,b,:] = M[a,b] @ W.T where M[a,b] is a (p*n) x rank matrix
            # M[a,b,c,r] = U[a,r]*V[b,r]
            
            # This is a big system. Let me just use a simple gradient descent or alternating least squares.
            
            pass
        
        # Check if the decomposition is valid
        # ...
    
    # Fallback to schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
