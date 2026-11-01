import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2x10 using 5 Strassen blocks
    # j=0,1 for A and B
    # k split into 5 blocks of 2
    for b in range(5):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # M1 = (A00 + A11)(B0k1 + B1k2)
        U[0][col] += 1
        U[3][col] += 1
        V[k1][col] += 1
        V[10 + k2][col] += 1
        # C0k1, C0k2, C1k1, C1k2 contributions
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes to C11, C22
        W[k1 * 2 + 0][col] += 1  # C0k1
        W[k2 * 2 + 1][col] += 1  # C1k2
        
        col += 1
        
        # M2 = (A10 + A11)B0k1
        U[3][col] += 1
        U[4][col] += 1  # Wait, A10 is row 1 col 0 -> index 1*3+0 = 3? No.
        # A is 2x3.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=0, j=2 -> 2
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        # i=1, j=2 -> 5
        
        # A10 is i=1, j=0 -> index 3.
        # A11 is i=1, j=1 -> index 4.
        # So U[3] and U[4].
        
        # B0k1 is j=0, k=k1 -> index 0*10 + k1 = k1.
        V[k1][col] += 1
        
        # C21 = M2 + M4 -> C1k1
        W[k1 * 2 + 1][col] += 1
        
        col += 1
        
        # M3 = A00(B1k1 - B1k2)
        U[0][col] += 1
        V[10 + k1][col] += 1
        V[10 + k2][col] -= 1
        
        # C12 = M3 + M7 -> C0k2
        W[k2 * 2 + 0][col] += 1
        
        col += 1
        
        # M4 = A11(B0k2 - B0k1)
        U[4][col] += 1
        V[k2][col] += 1
        V[k1][col] -= 1
        
        # C11 = M1 - M4 ... -> C0k1
        W[k1 * 2 + 0][col] -= 1
        # C21 = M2 + M4 -> C1k1
        W[k1 * 2 + 1][col] += 1
        
        col += 1
        
        # M5 = (A00 + A01)B1k2
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k2][col] += 1
        
        # C11 = M1 - M4 + M5 ... -> C0k1
        W[k1 * 2 + 0][col] += 1
        
        col += 1
        
        # M6 = (A10 - A00)(B0k1 + B0k2)
        U[3][col] += 1
        U[0][col] -= 1
        V[k1][col] += 1
        V[k2][col] += 1
        
        # C22 = M1 - M2 + M3 + M6 -> C1k2
        W[k2 * 2 + 1][col] += 1
        
        col += 1
        
        # M7 = (A01 - A11)(B1k1 + B1k2)
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k1][col] += 1
        V[10 + k2][col] += 1
        
        # C11 = M1 - M4 + M5 - M7 -> C0k1
        W[k1 * 2 + 0][col] -= 1
        # C12 = M3 + M7 -> C0k2
        W[k2 * 2 + 0][col] += 1
        
        col += 1

    # Part 2: 2x1x10 using j=2
    # A02 -> U[2], A12 -> U[5]
    # B2k -> V[20+k]
    for k in range(10):
        # i=0
        U[2][col] += 1
        V[20 + k][col] += 1
        W[k * 2 + 0][col] += 1
        col += 1
        
        # i=1
        U[5][col] += 1
        V[20 + k][col] += 1
        W[k * 2 + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
