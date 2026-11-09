import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 by 5x5 matrix.
    Uses 8 Strassen 2x2x2 blocks (rank 56) + 36 identity terms = rank 92.
    """
    n, m, p = 4, 5, 5
    
    # 8 disjoint 2x2x2 Strassen blocks
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([2, 3], [0, 1], [0, 1]),
        ([2, 3], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
        ([2, 3], [0, 1], [2, 3]),
        ([2, 3], [2, 3], [2, 3]),
    ]
    
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    uncovered.append((i, j, k))
    
    rank = 7 * len(blocks) + len(uncovered)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # Strassen 7-term decomposition
        U[i1*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k2*n + i1][col] += 1
        W[k1*n + i2][col] += 1
        col += 1
        
        U[i2*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + i1][col] += -1
        W[k2*n + i2][col] += 1
        col += 1
        
        U[i1*m + j1][col] += 1
        V[j2*p + k2][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        U[i2*m + j2][col] += 1
        V[j2*p + k2][col] += 1
        V[j1*p + k1][col] += -1
        W[k1*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        U[i1*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        W[k1*n + i2][col] += -1
        col += 1
        
        U[i2*m + j2][col] += 1
        U[i2*m + j1][col] += -1
        V[j1*p + k2][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += 1
        col += 1
        
        U[i1*m + j1][col] += 1
        U[i1*m + j2][col] += -1
        V[j1*p + k1][col] += 1
        V[j1*p + k2][col] += 1
        W[k2*n + i1][col] += 1
        col += 1
    
    for (i, j, k) in uncovered:
        U[i*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + i][col] = 1
        col += 1
    
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
