import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # A = [A1 | A2] where A1 is 2x2 (cols 0,1), A2 is 2x1 (col 2)
    # B = [B1; B2] where B1 is 2x9 (rows 0,1), B2 is 1x9 (row 2)
    # C = A1*B1 + A2*B2
    
    # Part 1: A1*B1 where A1 is 2x2, B1 is 2x9
    # Split B1 into 4 blocks of 2 cols + 1 block of 1 col
    # For each 2x2 x 2x2 block: Strassen (7 products)
    # For 2x2 x 2x1 block: schoolbook (4 products)
    
    for block in range(4):
        # B1 columns 2*block and 2*block+1
        bcol0 = 2 * block
        bcol1 = 2 * block + 1
        # Output columns bcol0 and bcol1
        
        # Strassen for 2x2 * 2x2:
        # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B_block = [[B[0,bcol0], B[0,bcol1]], [B[1,bcol0], B[1,bcol1]]]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,bcol0]+B[1,bcol1])
        U[0*3+0][col] = 1  # A[0,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*9+bcol0][col] = 1  # B[0,bcol0]
        V[1*9+bcol1][col] = 1  # B[1,bcol1]
        W[bcol0*2+0][col] = 1  # C[0,bcol0]
        W[bcol1*2+1][col] = 1  # C[1,bcol1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,bcol0]
        U[1*3+0][col] = 1  # A[1,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*9+bcol0][col] = 1  # B[0,bcol0]
        W[bcol0*2+1][col] = 1  # C[1,bcol0]
        W[bcol1*2+1][col] = 1  # C[1,bcol1]
        col += 1
        
        # M3 = A[0,0]*(B[0,bcol1]-B[1,bcol1])
        U[0*3+0][col] = 1  # A[0,0]
        V[0*9+bcol1][col] = 1  # B[0,bcol1]
        V[1*9+bcol1][col] = -1  # -B[1,bcol1]
        W[bcol1*2+0][col] = 1  # C[0,bcol1]
        W[bcol1*2+1][col] = 1  # C[1,bcol1]
        col += 1
        
        # M4 = A[1,1]*(B[1,bcol0]-B[0,bcol0])
        U[1*3+1][col] = 1  # A[1,1]
        V[1*9+bcol0][col] = 1  # B[1,bcol0]
        V[0*9+bcol0][col] = -1  # -B[0,bcol0]
        W[bcol0*2+0][col] = 1  # C[0,bcol0]
        W[bcol0*2+1][col] = 1  # C[1,bcol0]
        col += 1
        
        # M5 = (A[0,0]+A[1,0])*B[1,bcol1]
        U[0*3+0][col] = 1  # A[0,0]
        U[1*3+0][col] = 1  # A[1,0]
        V[1*9+bcol1][col] = 1  # B[1,bcol1]
        W[bcol1*2+0][col] = -1  # C[0,bcol1]
        W[bcol0*2+1][col] = 1  # C[1,bcol0]
        col += 1
        
        # M6 = (A[0,1]-A[1,0])*(B[0,bcol0]+B[1,bcol0])
        U[0*3+1][col] = 1  # A[0,1]
        U[1*3+0][col] = -1  # -A[1,0]
        V[0*9+bcol0][col] = 1  # B[0,bcol0]
        V[1*9+bcol0][col] = 1  # B[1,bcol0]
        W[bcol1*2+1][col] = 1  # C[1,bcol1]
        col += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[0,bcol1]+B[1,bcol1])
        U[0*3+1][col] = 1  # A[0,1]
        U[1*3+1][col] = -1  # -A[1,1]
        V[0*9+bcol1][col] = 1  # B[0,bcol1]
        V[1*9+bcol1][col] = 1  # B[1,bcol1]
        W[bcol0*2+0][col] = 1  # C[0,bcol0]
        W[bcol1*2+0][col] = 1  # C[0,bcol1]
        col += 1
    
    # 2x2 x 2x1 for B1 column 8:
    bcol = 8
    # C[0,bcol] = A[0,0]*B[0,bcol] + A[0,1]*B[1,bcol]
    # C[1,bcol] = A[1,0]*B[0,bcol] + A[1,1]*B[1,bcol]
    
    # Product 1: A[0,0]*B[0,bcol]
    U[0*3+0][col] = 1
    V[0*9+bcol][col] = 1
    W[bcol*2+0][col] = 1
    col += 1
    
    # Product 2: A[0,1]*B[1,bcol]
    U[0*3+1][col] = 1
    V[1*9+bcol][col] = 1
    W[bcol*2+0][col] = 1
    col += 1
    
    # Product 3: A[1,0]*B[0,bcol]
    U[1*3+0][col] = 1
    V[0*9+bcol][col] = 1
    W[bcol*2+1][col] = 1
    col += 1
    
    # Product 4: A[1,1]*B[1,bcol]
    U[1*3+1][col] = 1
    V[1*9+bcol][col] = 1
    W[bcol*2+1][col] = 1
    col += 1
    
    # Part 2: A2*B2 where A2 is 2x1 (col 2), B2 is 1x9 (row 2)
    # C[i,k] += A[1,2]*B[2,k] for i=0,1 and k=0..8
    # This is 18 products (schoolbook)
    for k in range(9):
        # C[0,k] += A[0,2]*B[2,k]
        U[0*3+2][col] = 1
        V[2*9+k][col] = 1
        W[k*2+0][col] = 1
        col += 1
        
        # C[1,k] += A[1,2]*B[2,k]
        U[1*3+2][col] = 1
        V[2*9+k][col] = 1
        W[k*2+1][col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
