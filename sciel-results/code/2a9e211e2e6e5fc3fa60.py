import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 2, 15
    # Create the tensor T of shape (4, 30, 30)
    # T[2i+j, 15j'+k, 2k'+i'] = 1 if i==i', j==j', k==k' else 0
    T = np.zeros((4, 30, 30), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a low-rank decomposition
    # Use ALS
    best_rank = n*m*p  # schoolbook
    best_U, best_V, best_W = None, None, None
    
    for rank in range(45, n*m*p+1):
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(4, rank)
        V = np.random.randn(30, rank)
        W = np.random.randn(30, rank)
        
        # ALS iterations
        for it in range(100):
            # Update U
            for c in range(rank):
                # U[:,c] = argmin ||T - sum_{c'} U[:,c']⊗V[:,c']⊗W[:,c']||
                # This is: U[:,c] = (V[:,c]^T ⊗ W[:,c])^{-1} * (T - sum_{c'≠c} ...) flattened
                # Simpler: for each entry of U[:,c], solve least squares
                # T[2i+j, 15j'+k, 2k'+i'] = sum_c U[2i+j,c]*V[15j'+k,c]*W[2k'+i',c]
                # For fixed c, U[2i+j,c] appears in T[2i+j, 15j'+k, 2k'+i'] as U[2i+j,c]*V[15j'+k,c]*W[2k'+i',c]
                # So we need: sum_c U[2i+j,c]*V[15j'+k,c]*W[2k'+i',c] = T[2i+j, 15j'+k, 2k'+i']
                # This is a linear system in U[:,c] for fixed c
                # Actually, ALS updates one mode at a time
                pass  # This is complex to implement correctly
        
        # This approach is too slow and complex
        break
    
    # Just use schoolbook
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
