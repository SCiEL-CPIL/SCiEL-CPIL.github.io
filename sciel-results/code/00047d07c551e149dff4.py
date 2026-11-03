import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Decomposition: C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each sub-product is 2x2 @ 2x7, schoolbook rank 14
    # Total rank = 28
    
    rank = 28
    U = np.zeros((n * m, rank))  # (8, 28)
    V = np.zeros((m * p, rank))  # (28, 28)
    W = np.zeros((p * n, rank))  # (14, 28)
    
    col = 0
    # First block: A[:,0:2] @ B[0:2,:]
    # A indices: i in [0,1], j in [0,1] -> U index i*4+j
    # B indices: j in [0,1], k in [0,6] -> V index j*7+k
    # C indices: i in [0,1], k in [0,6] -> W index k*2+i
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Second block: A[:,2:4] @ B[2:4,:]
    # A indices: i in [0,1], j in [2,3] -> U index i*4+j
    # B indices: j in [2,3], k in [0,6] -> V index j*7+k
    # C indices: i in [0,1], k in [0,6] -> W index k*2+i
    for i in range(n):
        for j in range(2, 4):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    assert col == rank
    # Check: sum_r U[:,r] outer V[:,r] should give the multiplication tensor
    # T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # In our factorization: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for valid (i,j,k)
    
    # Let's verify with a simple check
    # The tensor T has T[i,j,k] = 1 for all i in [0,1], j in [0,3], k in [0,6]
    # Our decomposition: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # For (i,j,k) with j in [0,1]: only first block contributes, and when r corresponds to (i,j,k), all three are 1
    # For (i,j,k) with j in [2,3]: only second block contributes
    # So the sum should be 1 for all valid (i,j,k)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
