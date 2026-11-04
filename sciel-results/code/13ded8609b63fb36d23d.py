import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U: (6, rank), V: (30, rank), W: (20, rank)
    
    # Rank 55 decomposition
    # 5 Strassen blocks (rank 7 each) + 1 Naive block (rank 20)
    rank = 5 * 7 + 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for A1 * B1
    # A1 is 2x2 (i in 0,1; j in 0,1)
    # B1 is 2x10 (j in 0,1; k in 0..9)
    # Partition B1 into 5 blocks of 2x2
    
    for t in range(5):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # B11 = B[0, k1], B12 = B[0, k2]
        # B21 = B[1, k1], B22 = B[1, k2]
        
        # M1 = (A11 + A22)(B11 + B22)
        U[0*3 + 0][col] = 1 # A11
        U[1*3 + 1][col] = 1 # A22
        V[0*10 + k1][col] = 1 # B11
        V[1*10 + k2][col] = 1 # B22
        W[k1*2 + 0][col] = 1 # C11 (i=0, k=k1)
        W[k2*2 + 1][col] = 1 # C22 (i=1, k=k2)
        col += 1
        
        # M2 = (A21 + A22)B11
        U[1*3 + 0][col] = 1 # A21
        U[1*3 + 1][col] = 1 # A22
        V[0*10 + k1][col] = 1 # B11
        W[k1*2 + 1][col] = 1 # C21 (i=1, k=k1)
        col += 1
        
        # M3 = A11(B12 - B22)
        U[0*3 + 0][col] = 1 # A11
        V[0*10 + k2][col] = 1 # B12
        V[1*10 + k2][col] = -1 # -B22
        W[k2*2 + 0][col] = 1 # C12 (i=0, k=k2)
        col += 1
        
        # M4 = A22(B21 - B11)
        U[1*3 + 1][col] = 1 # A22
        V[1*10 + k1][col] = 1 # B21
        V[0*10 + k1][col] = -1 # -B11
        W[k1*2 + 0][col] = 1 # C11 (i=0, k=k1)
        W[k2*2 + 1][col] = 1 # C22 (i=1, k=k2)
        col += 1
        
        # M5 = (A11 + A12)B22
        U[0*3 + 0][col] = 1 # A11
        U[0*3 + 1][col] = 1 # A12
        V[1*10 + k2][col] = 1 # B22
        W[k1*2 + 0][col] = -1 # C11 (i=0, k=k1)
        W[k2*2 + 1][col] = 1 # C22 (i=1, k=k2)
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[1*3 + 0][col] = 1 # A21
        U[0*3 + 0][col] = -1 # -A11
        V[0*10 + k1][col] = 1 # B11
        V[0*10 + k2][col] = 1 # B12
        W[k2*2 + 1][col] = 1 # C22 (i=1, k=k2)
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[0*3 + 1][col] = 1 # A12
        U[1*3 + 1][col] = -1 # -A22
        V[1*10 + k1][col] = 1 # B21
        V[1*10 + k2][col] = 1 # B22
        W[k1*2 + 0][col] = 1 # C11 (i=0, k=k1)
        col += 1

    # Part 2: Naive for A2 * B2
    # A2 is 2x1 (i in 0,1; j=2)
    # B2 is 1x10 (j=2; k in 0..9)
    # C2[i, k] = A[i, 2] * B[2, k]
    
    for i in range(2):
        for k in range(10):
            U[i*3 + 2][col] = 1 # A[i, 2]
            V[2*10 + k][col] = 1 # B[2, k]
            W[k*2 + i][col] = 1 # C[i, k]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
