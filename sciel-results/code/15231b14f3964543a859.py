import time
import numpy as np
import random

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition: rank = n*m*p = 52
    R0 = n * m * p
    U = [[0]*R0 for _ in range(nm)]
    V = [[0]*R0 for _ in range(mp)]
    W = [[0]*R0 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W, best_R = U, V, W, R0
    
    # Try to find a rank 3p+1 = 40 construction
    # For 2x2 x 2xp, rank is known to be 3p+1
    target_rank = 3 * p + 1  # 40
    
    if target_rank < best_R:
        U2 = [[0]*target_rank for _ in range(nm)]
        V2 = [[0]*target_rank for _ in range(mp)]
        W2 = [[0]*target_rank for _ in range(pn)]
        
        # Construction for rank 3p+1:
        # For each k, use 3 terms + 1 shared term
        # Term structure:
        # For k in range(p):
        #   col 3k:   U[0,0]=1, V[0,k]=1, W[k,0]=1
        #   col 3k+1: U[1,1]=1, V[1,k]=1, W[k,1]=1  
        #   col 3k+2: U[0,1]=1, V[1,k]=1, W[k,0]=1  -- wait, this conflicts
        # 
        # Let me use a different known construction:
        # P1_k = (A[0,0]+A[1,0]) * B[0,k]
        # P2_k = (A[0,1]+A[1,1]) * B[1,k]
        # P3_k = (A[0,0]-A[1,0]) * (B[0,k]-B[1,k])
        # 
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        #
        # From P1_k: A[0,0]*B[0,k] + A[1,0]*B[0,k]
        # From P2_k: A[0,1]*B[1,k] + A[1,1]*B[1,k]
        # From P3_k: A[0,0]*B[0,k] - A[0,0]*B[1,k] - A[1,0]*B[0,k] + A[1,0]*B[1,k]
        #
        # C[0,k] = (P1_k + P3_k)/2 + A[0,1]*B[1,k] - A[0,0]*B[1,k]/2 - A[1,0]*B[1,k]/2
        # This is getting too complex. Let me just use schoolbook.
        
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
