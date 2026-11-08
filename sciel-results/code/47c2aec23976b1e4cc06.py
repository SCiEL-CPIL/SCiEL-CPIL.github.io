import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block_idx in range(5):
        k1 = block_idx * 2
        k2 = k1 + 1
        
        # Strassen terms for top 2x2 of B (j=0,1)
        # m1
        U[0][col] += 1; U[4][col] += 1
        V[10*k1][col] += 1; V[10*k2 + 1][col] += 1
        W[2*k1][col] += 1; W[2*k2 + 1][col] += 1
        col += 1
        
        # m2
        U[3][col] += 1; U[4][col] += 1
        V[10*k1][col] += 1
        W[2*k1 + 1][col] += 1
        col += 1
        
        # m3
        U[0][col] += 1
        V[10*k2][col] += 1; V[10*k2 + 1][col] -= 1
        W[2*k2][col] += 1; W[2*k2 + 1][col] -= 1
        col += 1
        
        # m4
        U[4][col] += 1
        V[10*k1 + 1][col] += 1; V[10*k1][col] -= 1
        W[2*k1][col] -= 1; W[2*k1 + 1][col] += 1
        col += 1
        
        # m5
        U[0][col] += 1; U[1][col] += 1
        V[10*k2 + 1][col] += 1
        W[2*k2][col] += 1; W[2*k2 + 1][col] += 1
        col += 1
        
        # m6
        U[3][col] += 1; U[0][col] -= 1
        V[10*k1][col] += 1; V[10*k2][col] += 1
        W[2*k1][col] += 1; W[2*k2][col] -= 1
        col += 1
        
        # m7
        U[1][col] += 1; U[4][col] -= 1
        V[10*k1 + 1][col] += 1; V[10*k2 + 1][col] += 1
        W[2*k1 + 1][col] += 1; W[2*k2 + 1][col] -= 1
        col += 1
        
        # Extra 4 terms for j=2
        # a02 * b2k1
        U[2][col] += 1
        V[20 + k1][col] += 1
        W[2*k1][col] += 1
        col += 1
        
        # a02 * b2k2
        U[2][col] += 1
        V[20 + k2][col] += 1
        W[2*k2][col] += 1
        col += 1
        
        # a12 * b2k1
        U[5][col] += 1
        V[20 + k1][col] += 1
        W[2*k1 + 1][col] += 1
        col += 1
        
        # a12 * b2k2
        U[5][col] += 1
        V[20 + k2][col] += 1
        W[2*k2 + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
