"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    rank = 55
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=2 (Naive, Rank 20)
    # A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
            
    # Part 2: j in {0, 1} (Strassen, Rank 35)
    # Split k into 5 pairs
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    # Strassen patterns for 2x2 block
    # U indices: i*m + j (i in 0,1; j in 0,1)
    # V indices: j*p + k (j in 0,1; k in k1,k2)
    # W indices: k*n + i (k in k1,k2; i in 0,1)
    
    for k1, k2 in pairs:
        # M1 = (a+d)(z+w)
        U[0*m + 0][col] = 1.0
        U[1*m + 1][col] = 1.0
        V[1*p + k1][col] = 1.0
        V[1*p + k2][col] = 1.0
        W[k1*n + 0][col] = 1.0
        W[k2*n + 1][col] = 1.0
        col += 1
        
        # M2 = (c+d)w
        U[1*m + 0][col] = 1.0
        U[1*m + 1][col] = 1.0
        V[1*p + k2][col] = 1.0
        W[k2*n + 1][col] = -1.0
        col += 1
        
        # M3 = a(y-w)
        U[0*m + 0][col] = 1.0
        V[0*p + k2][col] = 1.0
        V[1*p + k2][col] = -1.0
        W[k2*n + 0][col] = 1.0
        W[k2*n + 1][col] = 1.0
        col += 1
        
        # M4 = d(x-z)
        U[1*m + 1][col] = 1.0
        V[0*p + k1][col] = 1.0
        V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = -1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # M5 = (a+b)z
        U[0*m + 0][col] = 1.0
        U[0*m + 1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k1*n + 0][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # M6 = (c-a)(x+z)
        U[1*m + 0][col] = 1.0
        U[0*m + 0][col] = -1.0
        V[0*p + k1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k1*n + 1][col] = -1.0
        W[k1*n + 0][col] = 1.0
        col += 1
        
        # M7 = (b-d)(y+w)
        U[0*m + 1][col] = 1.0
        U[1*m + 1][col] = -1.0
        V[0*p + k2][col] = 1.0
        V[1*p + k2][col] = 1.0
        W[k2*n + 0][col] = -1.0
        W[k2*n + 1][col] = 1.0
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
