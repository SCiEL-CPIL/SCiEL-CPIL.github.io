import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    # For k pairs (0,1), (2,3), (4,5): use Strassen for j=0,1 part
    for kp in range(3):
        k0 = kp * 2
        k1 = kp * 2 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0, col] = 1
        U[4, col] = 1
        V[0*p+k0, col] = 1
        V[1*p+k1, col] = 1
        W[k0*n+0, col] = 1
        W[k1*n+0, col] = 1
        W[k0*n+1, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3, col] = 1
        U[4, col] = 1
        V[0*p+k0, col] = 1
        W[k0*n+1, col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0, col] = 1
        V[0*p+k1, col] = 1
        V[1*p+k1, col] = -1
        W[k1*n+0, col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4, col] = 1
        V[1*p+k0, col] = 1
        V[0*p+k0, col] = -1
        W[k0*n+0, col] = 1
        W[k0*n+1, col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0, col] = 1
        U[1, col] = 1
        V[1*p+k1, col] = 1
        W[k0*n+0, col] = -1
        W[k1*n+1, col] = -1
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[3, col] = 1
        U[0, col] = -1
        V[0*p+k0, col] = 1
        V[0*p+k1, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1, col] = 1
        U[4, col] = -1
        V[1*p+k0, col] = 1
        V[1*p+k1, col] = 1
        W[k0*n+0, col] = 1
        W[k1*n+1, col] = -1
        col += 1
    
    # Handle j=2 for all k=0..6: schoolbook, rank 2*7=14
    for k in range(p):
        for i in range(n):
            U[i*m+2, col] = 1
            V[2*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    
    # Handle j=0,1 and k=6 (single k): schoolbook, rank 2*2=4
    for i in range(n):
        for j in range(2):
            U[i*m+j, col] = 1
            V[j*p+6, col] = 1
            W[6*n+i, col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
