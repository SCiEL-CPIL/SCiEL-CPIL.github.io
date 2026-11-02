import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0 (index 0 in 0-based, j=1 in 1-based)
    # 1x1x5 block
    for k in range(p):
        # C[i, k] += A[i, 0] * B[0, k]
        # i=0,1
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j=1
    # 1x1x5 block
    for k in range(p):
        for i in range(n):
            U[i * m + 1][col] = 1
            V[1 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 3: j=2
    # 1x1x5 block
    for k in range(p):
        for i in range(n):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 4: j=3,4 (indices 3,4 in 0-based)
    # 2x2x5 block using Strassen for cols 0-1, 2-3 and schoolbook for col 4
    
    # Sub-part 4a: Cols 0,1 (k=0,1)
    # Strassen 7 mults
    # M1 = (a14+a25)(b41+b52) -> U[0,3]=1, U[1,4]=1; V[3,0]=1, V[4,1]=1
    # Note: U index is i*m+j. 
    # i=0, j=3 -> idx 3. i=1, j=3 -> idx 8.
    # i=0, j=4 -> idx 4. i=1, j=4 -> idx 9.
    # V index is j*p+k.
    # j=3, k=0 -> idx 15. j=3, k=1 -> idx 16.
    # j=4, k=0 -> idx 20. j=4, k=1 -> idx 21.
    
    # M1
    U[0 * m + 3][col] = 1
    U[1 * m + 4][col] = 1
    V[3 * p + 0][col] = 1
    V[4 * p + 1][col] = 1
    W[0 * n + 0][col] = 1 # C[0,0] += M1
    W[1 * n + 1][col] = 1 # C[1,1] += M1
    col += 1
    
    # M2 = (a24+a25)b42
    U[1 * m + 3][col] = 1
    U[1 * m + 4][col] = 1
    V[3 * p + 1][col] = 1
    W[1 * n + 0][col] = 1 # C[1,0] += M2
    W[0 * n + 1][col] = 1 # C[0,1] += M2
    col += 1
    
    # M3 = a14(b51-b52)
    U[0 * m + 3][col] = 1
    V[4 * p + 0][col] = 1
    V[4 * p + 1][col] = -1
    W[0 * n + 0][col] = 1 # C[0,0] += M3
    W[1 * n + 1][col] = 1 # C[1,1] += M3
    col += 1
    
    # M4 = a25(b41-b51)
    U[1 * m + 4][col] = 1
    V[3 * p + 0][col] = 1
    V[4 * p + 0][col] = -1
    W[0 * n + 1][col] = 1 # C[0,1] += M4
    W[1 * n + 0][col] = 1 # C[1,0] += M4
    col += 1
    
    # M5 = (a14+a15)b52
    U[0 * m + 3][col] = 1
    U[0 * m + 4][col] = 1
    V[4 * p + 1][col] = 1
    W[0 * n + 0][col] = 1 # C[0,0] += M5
    W[1 * n + 1][col] = 1 # C[1,1] += M5
    col += 1
    
    # M6 = (a24-a14)(b41+b42)
    U[1 * m + 3][col] = 1
    U[0 * m + 3][col] = -1
    V[3 * p + 0][col] = 1
    V[3 * p + 1][col] = 1
    W[1 * n + 0][col] = 1 # C[1,0] += M6
    W[0 * n + 1][col] = 1 # C[0,1] += M6
    col += 1
    
    # M7 = (a15-a25)(b51+b52)
    U[0 * m + 4][col] = 1
    U[1 * m + 4][col] = -1
    V[4 * p + 0][col] = 1
    V[4 * p + 1][col] = 1
    W[0 * n + 0][col] = 1 # C[0,0] += M7
    W[1 * n + 1][col] = 1 # C[1,1] += M7
    col += 1
    
    # Sub-part 4b: Cols 2,3 (k=2,3)
    # Same structure, shift k indices
    
    # M8 = (a14+a25)(b43+b54)
    U[0 * m + 3][col] = 1
    U[1 * m + 4][col] = 1
    V[3 * p + 2][col] = 1
    V[4 * p + 3][col] = 1
    W[2 * n + 0][col] = 1 # C[0,2] += M8
    W[3 * n + 1][col] = 1 # C[1,3] += M8
    col += 1
    
    # M9 = (a24+a25)b44
    U[1 * m + 3][col] = 1
    U[1 * m + 4][col] = 1
    V[3 * p + 3][col] = 1
    W[3 * n + 0][col] = 1 # C[1,3] += M9
    W[2 * n + 1][col] = 1 # C[0,2] += M9
    col += 1
    
    # M10 = a14(b53-b54)
    U[0 * m + 3][col] = 1
    V[4 * p + 2][col] = 1
    V[4 * p + 3][col] = -1
    W[2 * n + 0][col] = 1 # C[0,2] += M10
    W[3 * n + 1][col] = 1 # C[1,3] += M10
    col += 1
    
    # M11 = a25(b43-b53)
    U[1 * m + 4][col] = 1
    V[3 * p + 2][col] = 1
    V[4 * p + 2][col] = -1
    W[2 * n + 1][col] = 1 # C[0,3] += M11
    W[3 * n + 0][col] = 1 # C[1,2] += M11
    col += 1
    
    # M12 = (a14+a15)b54
    U[0 * m + 3][col] = 1
    U[0 * m + 4][col] = 1
    V[4 * p + 3][col] = 1
    W[2 * n + 0][col] = 1 # C[0,2] += M12
    W[3 * n + 1][col] = 1 # C[1,3] += M12
    col += 1
    
    # M13 = (a24-a14)(b43+b44)
    U[1 * m + 3][col] = 1
    U[0 * m + 3][col] = -1
    V[3 * p + 2][col] = 1
    V[3 * p + 3][col] = 1
    W[3 * n + 0][col] = 1 # C[1,3] += M13
    W[2 * n + 1][col] = 1 # C[0,2] += M13
    col += 1
    
    # M14 = (a15-a25)(b53+b54)
    U[0 * m + 4][col] = 1
    U[1 * m + 4][col] = -1
    V[4 * p + 2][col] = 1
    V[4 * p + 3][col] = 1
    W[2 * n + 0][col] = 1 # C[0,2] += M14
    W[3 * n + 1][col] = 1 # C[1,3] += M14
    col += 1
    
    # Sub-part 4c: Col 4 (k=4)
    # Schoolbook 4 mults
    
    # M15 = a14*b45
    U[0 * m + 3][col] = 1
    V[3 * p + 4][col] = 1
    W[4 * n + 0][col] = 1 # C[0,4] += M15
    col += 1
    
    # M16 = a15*b55
    U[0 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 0][col] = 1 # C[0,4] += M16
    col += 1
    
    # M17 = a24*b45
    U[1 * m + 3][col] = 1
    V[3 * p + 4][col] = 1
    W[4 * n + 1][col] = 1 # C[1,4] += M17
    col += 1
    
    # M18 = a25*b55
    U[1 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 1][col] = 1 # C[1,4] += M18
    col += 1
    
    return {"U": U, "V": V, "W": W}
