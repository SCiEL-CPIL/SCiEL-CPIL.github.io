"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # 8 for A_right (2x1) * B_bot (1x4)
    # 7 for A_left (2x2) * B_top_left (2x2)
    # 7 for A_left (2x2) * B_top_right (2x2)
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_right * B_bot
    # A_right: i in {0,1}, j=2
    # B_bot: j=2, k in {0,1,2,3}
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_left * B_top_left
    # A_left: i in {0,1}, j in {0,1}
    # B_top_left: j in {0,1}, k in {0,1}
    # Strassen decomposition for 2x2x2
    # Variables:
    # A: a=A[0,0], b=A[0,1], d=A[1,0], e=A[1,1]
    # B: g=B[0,0], h=B[0,1], k=B[1,0], l=B[1,1]
    
    # M1 = (a+d)(g+k)
    U[0*m+0][col] = 1; U[1*m+0][col] = 1
    V[0*p+0][col] = 1; V[1*p+0][col] = 1
    # C11 += M1, C21 += M1
    W[0*n+0][col] = 1; W[0*n+1][col] = 1
    col += 1
    
    # M2 = (d+e)(k+l)
    U[1*m+0][col] = 1; U[1*m+1][col] = 1
    V[1*p+0][col] = 1; V[1*p+1][col] = 1
    # C11 -= M2, C12 += M2
    W[0*n+0][col] = -1; W[1*n+0][col] = 1
    col += 1
    
    # M3 = a(g-k)
    U[0*m+0][col] = 1
    V[0*p+0][col] = 1; V[1*p+0][col] = -1
    # C11 += M3, C21 -= M3
    W[0*n+0][col] = 1; W[0*n+1][col] = -1
    col += 1
    
    # M4 = e(l-h)
    U[1*m+1][col] = 1
    V[1*p+1][col] = 1; V[0*p+1][col] = -1
    # C12 -= M4, C22 += M4
    W[1*n+0][col] = -1; W[1*n+1][col] = 1
    col += 1
    
    # M5 = (a+b)(k+l)
    U[0*m+0][col] = 1; U[0*m+1][col] = 1
    V[1*p+0][col] = 1; V[1*p+1][col] = 1
    # C22 += M5
    W[1*n+1][col] = 1
    col += 1
    
    # M6 = (d+a)(g-k)
    U[1*m+0][col] = 1; U[0*m+0][col] = 1
    V[0*p+0][col] = 1; V[1*p+0][col] = -1
    # C21 += M6
    W[0*n+1][col] = 1
    col += 1
    
    # M7 = (e+b)(h-l)
    U[1*m+1][col] = 1; U[0*m+1][col] = 1
    V[0*p+1][col] = 1; V[1*p+1][col] = -1
    # C12 += M7
    W[1*n+0][col] = 1
    col += 1

    # Part 3: A_left * B_top_right
    # A_left: i in {0,1}, j in {0,1}
    # B_top_right: j in {0,1}, k in {2,3}
    
    # M1 = (a+d)(g+k) -> g=B[0,2], k=B[1,2]
    U[0*m+0][col] = 1; U[1*m+0][col] = 1
    V[0*p+2][col] = 1; V[1*p+2][col] = 1
    # C13 += M1, C23 += M1
    W[2*n+0][col] = 1; W[2*n+1][col] = 1
    col += 1
    
    # M2 = (d+e)(k+l) -> k=B[1,2], l=B[1,3]
    U[1*m+0][col] = 1; U[1*m+1][col] = 1
    V[1*p+2][col] = 1; V[1*p+3][col] = 1
    # C13 -= M2, C14 += M2
    W[2*n+0][col] = -1; W[3*n+0][col] = 1
    col += 1
    
    # M3 = a(g-k) -> g=B[0,2], k=B[1,2]
    U[0*m+0][col] = 1
    V[0*p+2][col] = 1; V[1*p+2][col] = -1
    # C13 += M3, C23 -= M3
    W[2*n+0][col] = 1; W[2*n+1][col] = -1
    col += 1
    
    # M4 = e(l-h) -> l=B[1,3], h=B[0,3]
    U[1*m+1][col] = 1
    V[1*p+3][col] = 1; V[0*p+3][col] = -1
    # C14 -= M4, C24 += M4
    W[3*n+0][col] = -1; W[3*n+1][col] = 1
    col += 1
    
    # M5 = (a+b)(k+l) -> k=B[1,2], l=B[1,3]
    U[0*m+0][col] = 1; U[0*m+1][col] = 1
    V[1*p+2][col] = 1; V[1*p+3][col] = 1
    # C24 += M5
    W[3*n+1][col] = 1
    col += 1
    
    # M6 = (d+a)(g-k) -> g=B[0,2], k=B[1,2]
    U[1*m+0][col] = 1; U[0*m+0][col] = 1
    V[0*p+2][col] = 1; V[1*p+2][col] = -1
    # C23 += M6
    W[2*n+1][col] = 1
    col += 1
    
    # M7 = (e+b)(h-l) -> h=B[0,3], l=B[1,3]
    U[1*m+1][col] = 1; U[0*m+1][col] = 1
    V[0*p+3][col] = 1; V[1*p+3][col] = -1
    # C14 += M7
    W[3*n+0][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
