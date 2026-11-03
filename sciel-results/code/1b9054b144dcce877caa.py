import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part for A[0:2, 0:2] and B[0:2, :]
    # B columns are processed in pairs: (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # Term 1: M1 = (A00 + A11)(B0k1 + B1k2)
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + k1][col] += 1
        V[1 * p + k2][col] += 1
        W[k1 * n + 0][col] += 1
        W[k2 * n + 1][col] += 1
        col += 1
        
        # Term 2: M2 = (A10 + A11)B0k1
        U[1 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + k1][col] += 1
        W[k1 * n + 1][col] += 1
        W[k2 * n + 1][col] -= 1
        col += 1
        
        # Term 3: M3 = A00 B1k2
        U[0 * m + 0][col] += 1
        V[1 * p + k2][col] += 1
        W[k2 * n + 0][col] += 1
        W[k2 * n + 1][col] += 1
        col += 1
        
        # Term 4: M4 = A11 (B1k2 - B0k1)
        U[1 * m + 1][col] += 1
        V[1 * p + k2][col] += 1
        V[0 * p + k1][col] -= 1
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # Term 5: M5 = (A00 + A01)B1k2
        U[0 * m + 0][col] += 1
        U[0 * m + 1][col] += 1
        V[1 * p + k2][col] += 1
        W[k1 * n + 0][col] -= 1
        W[k2 * n + 0][col] += 1
        col += 1
        
        # Term 6: M6 = (A11 - A00)(B0k1 + B1k1)
        U[1 * m + 1][col] += 1
        U[0 * m + 0][col] -= 1
        V[0 * p + k1][col] += 1
        V[1 * p + k1][col] += 1
        W[k2 * n + 1][col] += 1
        col += 1
        
        # Term 7: M7 = (A01 - A11)(B1k2 + B1k1)
        U[0 * m + 1][col] += 1
        U[1 * m + 1][col] -= 1
        V[1 * p + k2][col] += 1
        V[1 * p + k1][col] += 1
        W[k1 * n + 0][col] += 1
        col += 1

    # Linear parts
    # i=0, j=2, k=0..7
    for k in range(p):
        U[0 * m + 2][col] += 1
        V[2 * p + k][col] += 1
        W[k * n + 0][col] += 1
        col += 1
        
    # i=1, j=2, k=0..7
    for k in range(p):
        U[1 * m + 2][col] += 1
        V[2 * p + k][col] += 1
        W[k * n + 1][col] += 1
        col += 1
        
    # i=2, j=0, k=0..7
    for k in range(p):
        U[2 * m + 0][col] += 1
        V[0 * p + k][col] += 1
        W[k * n + 2][col] += 1
        col += 1
        
    # i=2, j=1, k=0..7
    for k in range(p):
        U[2 * m + 1][col] += 1
        V[1 * p + k][col] += 1
        W[k * n + 2][col] += 1
        col += 1
        
    # i=2, j=2, k=0..7
    for k in range(p):
        U[2 * m + 2][col] += 1
        V[2 * p + k][col] += 1
        W[k * n + 2][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
