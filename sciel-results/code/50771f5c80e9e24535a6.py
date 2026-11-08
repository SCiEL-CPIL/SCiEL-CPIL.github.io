"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal (known) exact tensor decomposition for 
    multiplying a 3x4 matrix by a 4x4 matrix.
    
    The standard schoolbook decomposition has rank n*m*p = 3*4*4 = 48.
    It is defined by U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 
    for each term c corresponding to (i,j,k).
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave margin for overhead
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Schoolbook term for C[i,k] += A[i,j] * B[j,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (official trace convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                
                column += 1
                
                # Deadline check (unlikely to trigger for rank 48, but required)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
