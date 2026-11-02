import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 68 decomposition
    # A is 3x3, B is 3x8, C is 3x8
    # Split A into 2x2, 2x1, 1x2, 1x1
    # Split B into 2x8, 1x8
    # B_top (2x8) split into 4 blocks of 2x2
    # B_bot (1x8) split into 8 blocks of 1x1
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A11 * B_top (2x2 * 2x8) -> 4 blocks of 2x2 * 2x2 (Strassen)
    # A11 indices in A: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    # B_top indices in B: rows 0,1. cols 0..7.
    # Block b (0..3): cols 2b, 2b+1.
    for b in range(4):
        # Strassen for 2x2 * 2x2
        # A11 is fixed indices in U
        # B_block is fixed indices in V
        # C_block is fixed indices in W
        
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to map these to the global indices.
        # This is getting complicated to hardcode.
        # Let's just use schoolbook for the 2x2*2x2 blocks?
        # Rank 4 for each. 4*4=16.
        # Total 16 + 16 + 16 + 8 = 56?
        # Wait, if I use schoolbook for 2x2*2x2, rank is 4.
        # 4 blocks * 4 = 16.
        # Previous estimate with Strassen was 28.
        # 16 is better!
        
        # Let's re-verify rank of 2x2*2x2.
        # Schoolbook is 4. Strassen is 7.
        # So schoolbook is better here!
        
        # So A11 * B_top rank is 16.
        
        for i in range(2): # rows of A11 (0,1)
            for j in range(2): # cols of A11 / rows of B_top (0,1)
                for k in range(2): # cols of B_block (0,1)
                    a_idx = i * m + j # i is 0,1 (row of A), j is 0,1 (col of A)
                    # A is 3x3. A11 is top-left 2x2.
                    # A[i,j] is at index i*3+j.
                    # i in 0,1. j in 0,1.
                    
                    b_idx = j * p + (2*b + k) # j is row of B_top (0,1). col is 2b+k.
                    # B is 3x8. B_top is rows 0,1.
                    
                    c_idx = (2*b + k) * n + i # col of C is 2b+k. row of C is i.
                    # C is 3x8. C_top is rows 0,1.
                    
                    U[a_idx][col] = 1
                    V[b_idx][col] = 1
                    W[c_idx][col] = 1
                    col += 1
                    
    # 2. A12 * B_bot (2x1 * 1x8)
    # A12 is rows 0,1 col 2 of A.
    # B_bot is row 2 of B. cols 0..7.
    # C_bot? No, C_top?
    # C = A * B.
    # C_top (rows 0,1) = A11 B_top + A12 B_bot.
    # So A12 B_bot contributes to C_top.
    # A12 is 2x1. B_bot is 1x8.
    # C_block is 2x8.
    for i in range(2): # rows of A12 (0,1)
        for k in range(8): # cols of B_bot (0..7)
            a_idx = i * m + 2 # i in 0,1. col 2.
            b_idx = 2 * p + k # row 2. col k.
            c_idx = k * n + i # col k. row i.
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
            
    # 3. A21 * B_top (1x2 * 2x8)
    # A21 is row 2, cols 0,1 of A.
    # B_top is rows 0,1, cols 0..7.
    # C_bot (row 2) = A21 B_top + A22 B_bot.
    # So A21 B_top contributes to C_bot.
    for j in range(2): # cols of A21 / rows of B_top (0,1)
        for k in range(8): # cols of B_top (0..7)
            a_idx = 2 * m + j # row 2. col j.
            b_idx = j * p + k # row j. col k.
            c_idx = k * n + 2 # col k. row 2.
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
            
    # 4. A22 * B_bot (1x1 * 1x8)
    # A22 is row 2, col 2 of A.
    # B_bot is row 2, cols 0..7.
    # C_bot (row 2) += A22 B_bot.
    for k in range(8):
        a_idx = 2 * m + 2 # row 2, col 2.
        b_idx = 2 * p + k # row 2, col k.
        c_idx = k * n + 2 # col k, row 2.
        
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[c_idx][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
