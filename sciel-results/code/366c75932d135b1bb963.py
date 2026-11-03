import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 44 decomposition
    # Split p=8 into two blocks of 4
    # For each block, split m=3 into 2 and 1
    # For the 2x2 part, use Strassen (rank 7)
    # For the 2x1 part, use schoolbook (rank 8)
    
    # Total rank = 2 * (2 * 7 + 8) = 2 * 22 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # We have two blocks for p: 0-3 and 4-7
    for block in range(2):
        # Block offset in p
        p_offset = block * 4
        
        # Part 1: A_1 (2x2) * B_1 (2x4)
        # B_1 is split into two 2x2 matrices
        for sub_block in range(2):
            p_sub_offset = p_offset + sub_block * 2
            
            # Strassen decomposition for 2x2x2
            # Indices: i in 0..1, j in 0..1, k in 0..1
            # Strassen:
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # U entries for A_1 (rows 0,1; cols 0,1)
            # V entries for B_1 (rows 0,1; cols 0,1)
            # W entries for C_1 (rows 0,1; cols 0,1)
            
            # M1
            U[0*m+0][col] = 1; U[1*m+1][col] = 1
            V[0*p+0][col] = 1; V[1*p+1][col] = 1
            W[0*n+0][col] = 1; W[1*n+1][col] = 1
            W[0*n+1][col] = -1; W[1*n+0][col] = -1
            col += 1
            
            # M2
            U[1*m+0][col] = 1; U[1*m+1][col] = 1
            V[0*p+0][col] = 1
            W[0*n+0][col] = 1; W[1*n+0][col] = 1
            col += 1
            
            # M3
            U[0*m+0][col] = 1
            V[0*p+1][col] = 1; V[1*p+1][col] = -1
            W[1*n+1][col] = 1
            col += 1
            
            # M4
            U[1*m+1][col] = 1
            V[1*p+0][col] = 1; V[0*p+0][col] = -1
            W[0*n+0][col] = 1; W[1*n+0][col] = 1
            col += 1
            
            # M5
            U[0*m+0][col] = 1; U[0*m+1][col] = 1
            V[1*p+1][col] = 1
            W[1*n+1][col] = 1
            col += 1
            
            # M6
            U[1*m+0][col] = 1; U[0*m+0][col] = -1
            V[0*p+0][col] = 1; V[0*p+1][col] = 1
            W[0*n+1][col] = 1
            col += 1
            
            # M7
            U[0*m+1][col] = 1; U[1*m+1][col] = -1
            V[1*p+0][col] = 1; V[1*p+1][col] = 1
            W[1*n+1][col] = 1
            col += 1
            
        # Part 2: A_2 (2x1) * B_2 (1x4)
        # A_2 is row 2 of A (index 2)
        # B_2 is row 2 of B (index 2)
        # C_2 is the contribution to C from A_2 B_2
        # C_{i,k} += A_{i,2} B_{2,k}
        # Rank 8
        for i in range(n):
            for k in range(4):
                U[i*m+2][col] = 1
                V[2*p + (p_sub_offset + k)][col] = 1 # Wait, B_2 is 1x4, so rows 0..3 of B_2 correspond to cols 0..3 of B_X
                # B_X is 2x4. B_2 is the second row of B_X?
                # No, B_X is 3x4. B_1 is first 2 rows, B_2 is last 1 row.
                # So B_2 is row 2 of B_X.
                # In the global B, row 2 is index 2.
                # Cols are p_offset + k.
                V[2*p + (p_offset + k)][col] = 1
                W[(p_offset + k)*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
