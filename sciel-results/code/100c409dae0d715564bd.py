import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 51 decomposition
    # A (2x3) * B (3x9) = C (2x9)
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (2x9) and B2 (1x9)
    # C = A1*B1 + A2*B2
    
    # A2*B2 is 2x1x9. Rank 18.
    # A1*B1 is 2x2x9.
    # Split B1 into 3 blocks of 2x3.
    # Each 2x2x3 block:
    # Split 3 into 2+1.
    # 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = 11.
    # Total for A1*B1: 3 * 11 = 33.
    # Total rank: 33 + 18 = 51.
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)] # 6 x 51
    V = [[0] * rank for _ in range(m * p)] # 27 x 51
    W = [[0] * rank for _ in range(p * n)] # 18 x 51
    
    col = 0
    
    # Part 1: A2 * B2 (2x1x9)
    # A2 is col 2 of A (index 2 in 0-2). Row 0,1 of A.
    # A indices: i*m+j. i in 0,1. j=2.
    # A[0,2] is U[0*3+2] = U[2]
    # A[1,2] is U[1*3+2] = U[5]
    # B2 is row 2 of B (index 2). Cols 0-8.
    # B[2,k] is V[2*9+k] = V[18+k]
    # C[i,k] is W[k*2+i]
    
    for k in range(9):
        U[2][col] = 1  # A[0,2]
        V[18+k][col] = 1 # B[2,k]
        W[k*2+0][col] = 1 # C[0,k]
        col += 1
        
        U[5][col] = 1  # A[1,2]
        V[18+k][col] = 1 # B[2,k]
        W[k*2+1][col] = 1 # C[1,k]
        col += 1
        
    # Part 2: A1 * B1 (2x2x9)
    # A1 is cols 0,1 of A.
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B1 is rows 0,1 of B.
    # B[0,k]=V[k], B[1,k]=V[9+k]
    
    # Split B1 into 3 blocks of 3 cols.
    for block in range(3):
        k_start = block * 3
        # Block is 2x3.
        # Split into 2x2 and 2x1.
        
        # 2x2 part: cols k_start, k_start+1
        # Strassen 2x2x2
        # U indices: 0,1,3,4
        # V indices: k_start, k_start+1, 9+k_start, 9+k_start+1
        # W indices: (k_start)*2+0, (k_start)*2+1, (k_start+1)*2+0, (k_start+1)*2+1
        
        # M1 = (a00+a11)(b00+b11)
        U[0][col] = 1; U[4][col] = 1
        V[k_start][col] = 1; V[9+k_start+1][col] = 1
        W[k_start*2+0][col] = 1; W[(k_start+1)*2+1][col] = 1
        W[k_start*2+1][col] = -1; W[(k_start+1)*2+0][col] = -1
        col += 1
        
        # M2 = (a10+a11)b00
        U[3][col] = 1; U[4][col] = 1
        V[k_start][col] = 1
        W[k_start*2+0][col] = 1; W[(k_start+1)*2+0][col] = 1
        col += 1
        
        # M3 = a00(b11-b01)
        U[0][col] = 1
        V[9+k_start+1][col] = 1; V[k_start+1][col] = -1
        W[(k_start+1)*2+1][col] = 1; W[k_start*2+1][col] = -1
        col += 1
        
        # M4 = a11(b10-b00)
        U[4][col] = 1
        V[9+k_start][col] = 1; V[k_start][col] = -1
        W[(k_start+1)*2+0][col] = 1; W[k_start*2+0][col] = -1
        col += 1
        
        # M5 = (a00+a01)b11
        U[0][col] = 1; U[1][col] = 1
        V[9+k_start+1][col] = 1
        W[k_start*2+1][col] = 1; W[(k_start+1)*2+1][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[3][col] = 1; U[0][col] = -1
        V[k_start][col] = 1; V[k_start+1][col] = 1
        W[k_start*2+0][col] = 1; W[(k_start+1)*2+0][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1][col] = 1; U[4][col] = -1
        V[9+k_start][col] = 1; V[9+k_start+1][col] = 1
        W[k_start*2+1][col] = 1; W[(k_start+1)*2+1][col] = 1
        col += 1
        
        # 2x1 part: col k_start+2
        # Schoolbook 2x2x1
        # C0 = a00*b0 + a01*b1
        # C1 = a10*b0 + a11*b1
        # 4 mults
        
        # a00*b0
        U[0][col] = 1
        V[k_start+2][col] = 1
        W[(k_start+2)*2+0][col] = 1
        col += 1
        
        # a01*b1
        U[1][col] = 1
        V[9+k_start+2][col] = 1
        W[(k_start+2)*2+0][col] = 1
        col += 1
        
        # a10*b0
        U[3][col] = 1
        V[k_start+2][col] = 1
        W[(k_start+2)*2+1][col] = 1
        col += 1
        
        # a11*b1
        U[4][col] = 1
        V[9+k_start+2][col] = 1
        W[(k_start+2)*2+1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
