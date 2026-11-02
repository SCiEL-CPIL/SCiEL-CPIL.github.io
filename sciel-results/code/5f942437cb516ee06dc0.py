import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 multiplication using Block Strassen (Rank 56)
    # Part 2: 4x4x2 multiplication using Schoolbook (Rank 32)
    # Total Rank: 56 + 32 = 88
    
    rank_part1 = 56
    rank_part2 = 32
    total_rank = rank_part1 + rank_part2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Block indices for 4x4
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # We need to compute 8 products:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # For each product Aij Bkl, we use 7 multiplications.
    # Total 8 * 7 = 56 multiplications.
    
    col = 0
    for ci in range(2): # C row block
        for cj in range(2): # C col block
            # Determine which A and B blocks contribute
            # C[ci, cj] = sum_{k} A[ci, k] B[k, cj]
            # Here k is the block index 0 or 1
            for k in range(2):
                ai = ci
                aj = k
                bi = k
                bj = cj
                # Product A[ai, aj] * B[bi, bj]
                # Apply Strassen to this 2x2x2 product
                # A_block is 2x2, B_block is 2x2
                # A_block indices in A: rows 2*ai to 2*ai+1, cols 2*aj to 2*aj+1
                # B_block indices in B: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
                
                # Strassen terms
                # We define the 7 terms for this specific block product
                # Term 1: (A00+A11)(B00+B11)
                # In global indices:
                # A00 is A[2*ai, 2*aj], A11 is A[2*ai+1, 2*aj+1]
                # B00 is B[2*bi, 2*bj], B11 is B[2*bi+1, 2*bj+1]
                
                # We will add 7 columns for this product
                for s in range(7):
                    # s=0: M1
                    if s == 0:
                        # A parts: A00, A11
                        U[2*ai * m + 2*aj][col] = 1
                        U[(2*ai+1) * m + (2*aj+1)][col] = 1
                        # B parts: B00, B11
                        V[2*bi * p + 2*bj][col] = 1
                        V[(2*bi+1) * p + (2*bj+1)][col] = 1
                        # C parts: C00, C01, C10, C11 (relative to block)
                        # M1 contributes to C00, C11
                        # C00 relative: (2*ci, 2*cj)
                        # C11 relative: (2*ci+1, 2*cj+1)
                        W[(2*cj) * n + (2*ci)][col] = 1
                        W[(2*cj+1) * n + (2*ci+1)][col] = 1
                    elif s == 1:
                        # M2: (A10+A11)B00
                        U[(2*ai+1) * m + 2*aj][col] = 1
                        U[(2*ai+1) * m + (2*aj+1)][col] = 1
                        V[2*bi * p + 2*bj][col] = 1
                        # C parts: C10, C11
                        W[(2*cj) * n + (2*ci+1)][col] = 1
                        W[(2*cj+1) * n + (2*ci+1)][col] = 1
                    elif s == 2:
                        # M3: A00(B01-B11)
                        U[2*ai * m + 2*aj][col] = 1
                        V[2*bi * p + (2*bj+1)][col] = 1
                        V[(2*bi+1) * p + (2*bj+1)][col] = -1
                        # C parts: C01, C11
                        W[(2*cj+1) * n + (2*ci)][col] = 1
                        W[(2*cj+1) * n + (2*ci+1)][col] = 1
                    elif s == 3:
                        # M4: A11(B10-B00)
                        U[(2*ai+1) * m + (2*aj+1)][col] = 1
                        V[(2*bi+1) * p + 2*bj][col] = 1
                        V[2*bi * p + 2*bj][col] = -1
                        # C parts: C00, C10
                        W[(2*cj) * n + (2*ci)][col] = 1
                        W[(2*cj) * n + (2*ci+1)][col] = 1
                    elif s == 4:
                        # M5: (A00+A01)B11
                        U[2*ai * m + 2*aj][col] = 1
                        U[2*ai * m + (2*aj+1)][col] = 1
                        V[(2*bi+1) * p + (2*bj+1)][col] = 1
                        # C parts: C00, C01
                        W[(2*cj) * n + (2*ci)][col] = -1
                        W[(2*cj+1) * n + (2*ci)][col] = 1
                    elif s == 5:
                        # M6: (A10-A00)(B00+B01)
                        U[(2*ai+1) * m + 2*aj][col] = 1
                        U[2*ai * m + 2*aj][col] = -1
                        V[2*bi * p + 2*bj][col] = 1
                        V[2*bi * p + (2*bj+1)][col] = 1
                        # C parts: C11
                        W[(2*cj+1) * n + (2*ci+1)][col] = 1
                    elif s == 6:
                        # M7: (A01-A11)(B10+B11)
                        U[2*ai * m + (2*aj+1)][col] = 1
                        U[(2*ai+1) * m + (2*aj+1)][col] = -1
                        V[(2*bi+1) * p + 2*bj][col] = 1
                        V[(2*bi+1) * p + (2*bj+1)][col] = 1
                        # C parts: C00, C11
                        W[(2*cj) * n + (2*ci)][col] = 1
                        W[(2*cj+1) * n + (2*ci+1)][col] = -1
                        
                    col += 1
    
    # Part 2: 4x4x2 Schoolbook
    # Columns 4 and 5 of B
    for i in range(n):
        for j in range(m):
            for k in range(2):
                # k=0 corresponds to col 4 of B, k=1 to col 5
                b_col = 4 + k
                U[i * m + j][col] = 1
                V[j * p + b_col][col] = 1
                W[b_col * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
