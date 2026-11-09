import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # Part 1: A_1 (2x2) * B_1 (2x6) -> 3 blocks of 2x2x2, each rank 7
    # Part 2: A_2 (2x1) * B_2 (1x6) -> rank 12
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We have 3 blocks of B, so 3 sets of 7 columns
    for block in range(3):
        # Block uses B cols 2*block, 2*block+1
        # A cols 0, 1
        # C cols 2*block, 2*block+1
        
        # P1
        U[0*2+0][col] = 1; U[1*2+1][col] = 1
        V[0*6+2*block][col] = 1; V[1*6+2*block+1][col] = 1
        # W for C00 and C11
        W[2*block*2+0][col] = 1 # k=2b, i=0
        W[(2*block+1)*2+1][col] = 1 # k=2b+1, i=1
        col += 1
        
        # P2
        U[1*2+0][col] = 1; U[1*2+1][col] = 1
        V[0*6+2*block][col] = 1
        W[2*block*2+1][col] = 1 # k=2b, i=1
        col += 1
        
        # P3
        U[0*2+0][col] = 1
        V[0*6+2*block+1][col] = 1; V[1*6+2*block+1][col] = -1
        W[2*block*2+0][col] = 1 # k=2b, i=0
        W[(2*block+1)*2+0][col] = 1 # k=2b+1, i=0
        col += 1
        
        # P4
        U[1*2+1][col] = 1
        V[1*6+2*block][col] = 1; V[0*6+2*block][col] = -1
        W[2*block*2+0][col] = 1 # k=2b, i=0
        W[2*block*2+1][col] = 1 # k=2b, i=1
        col += 1
        
        # P5
        U[0*2+0][col] = 1; U[0*2+1][col] = 1
        V[1*6+2*block+1][col] = 1
        W[2*block*2+0][col] = -1 # k=2b, i=0
        W[(2*block+1)*2+0][col] = 1 # k=2b+1, i=0
        col += 1
        
        # P6
        U[1*2+0][col] = 1; U[0*2+0][col] = -1
        V[0*6+2*block][col] = 1; V[0*6+2*block+1][col] = 1
        W[(2*block+1)*2+1][col] = 1 # k=2b+1, i=1
        col += 1
        
        # P7
        U[0*2+1][col] = 1; U[1*2+1][col] = -1
        V[1*6+2*block][col] = 1; V[1*6+2*block+1][col] = 1
        W[2*block*2+0][col] = 1 # k=2b, i=0
        col += 1

    # Part 2: A_2 (col 2) * B_2 (row 2)
    # C[i,k] = A[i,2] B[2,k]
    # Need 12 terms
    for i in range(2):
        for k in range(6):
            U[i*2+2][col] = 1
            V[2*6+k][col] = 1
            W[k*2+i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
