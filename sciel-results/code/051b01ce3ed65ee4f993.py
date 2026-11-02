"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Verify schoolbook
    # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Try to reduce rank using SVD-based approach
    # The idea: if we have a decomposition, we can try to find a lower rank one
    # by exploiting linear dependencies
    
    # For this specific tensor, let's try a structured approach
    # M_{2,2,11}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # One approach: for each k, we have a 2x2 matrix multiplication
    # But we can't easily share across k values
    
    # Let's try: can we reduce by noting that the B matrix is 2x11?
    # The 11 columns of B are independent, so we likely need all 11*2 = 22 
    # "B slots" times the 2 A-rows = 44 terms minimum? Not necessarily.
    
    # Let me try a numerical rank detection approach
    # Reshape T in different ways and compute SVD
    
    # T as (nm x mp) for each w-index: not directly useful
    # Better: the tensor rank is at least the rank of any matricization
    
    # Matricization 1: (nm x mp*pn)
    T1 = T.reshape(n * m, m * p * p * n)
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    
    # Matricization 2: (mp x nm*pn)  
    T2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    
    # Matricization 3: (pn x nm*mp)
    T3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    r3 = np.linalg.matrix_rank(T3, tol=1e-10)
    
    # The tensor rank is at least max(r1, r2, r3)
    # For our tensor, let's see what these are
    
    # If the lower bound is 44, then schoolbook is optimal
    # If lower, we can try to find a better decomposition
    
    # For now, return schoolbook as it's guaranteed correct
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
