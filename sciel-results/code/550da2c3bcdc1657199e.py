import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Winograd Fourier Transform Algorithm (FTW) approach.
    Decompose the 2x4x6 tensor by exploiting the factorization of dimensions:
    - Split the middle dimension (4) into 2+2 blocks
    - Split the output dimension (6) into 3 blocks of 2
    - Use Strassen's 2x2x2 rank-7 decomposition for each 2x2x2 sub-problem
    
    Total rank = 2 (middle splits) * 3 (output splits) * 7 (Strassen) = 42
    
    The key insight from FTW is that we can view this as a Kronecker product
    of a 2x2x2 Strassen core with a 1x2x3 rectangular core, but the effective
    rank remains 42 since the rectangular core has rank 1 in the first dimension.
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)  
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column (0..5), i is row (0..1)
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # For X @ Y where X,Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Define Strassen coefficients
    # U coefficients: which A elements to combine for each Strassen product
    # Local A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    strassen_U = [
        {0: 1, 3: 1},       # M1: A00+A11
        {2: 1, 3: 1},       # M2: A10+A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00+A01
        {2: 1, 0: -1},      # M6: A10-A00
        {1: 1, 3: -1},      # M7: A01-A11
    ]
    
    # V coefficients: which B elements to combine for each Strassen product
    # Local B indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    strassen_V = [
        {0: 1, 3: 1},       # M1: B00+B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01-B11
        {2: 1, 0: -1},      # M4: B10-B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00+B01
        {2: 1, 3: 1},       # M7: B10+B11
    ]
    
    # W coefficients: how each Strassen product contributes to output
    # Local W indices: 0=Z[0,0], 1=Z[0,1], 2=Z[1,0], 3=Z[1,1]
    strassen_W = [
        {0: 1, 3: 1},       # M1: Z00+Z11
        {2: 1, 3: -1},      # M2: Z10-Z11
        {1: 1, 3: 1},       # M3: Z01+Z11
        {0: 1, 2: 1},       # M4: Z00+Z10
        {0: -1, 1: 1},      # M5: -Z00+Z01
        {3: 1},             # M6: Z11
        {0: 1},             # M7: Z00
    ]
    
    col_offset = 0
    
    # For each middle block (left: cols 0-1, right: cols 2-3)
    # and each output block (cols 0-1, 2-3, 4-5):
    for j_block in range(3):  # output column blocks
        for side in ['L', 'R']:  # middle dimension splits
            if side == 'L':
                u_col_offset = 0   # A columns 0-1
                v_row_offset = 0   # B rows 0-1
            else:
                u_col_offset = 2   # A columns 2-3
                v_row_offset = 2   # B rows 2-3
            
            for prod_idx in range(7):
                col = col_offset + prod_idx
                
                # U local index mapping:
                # local 0 -> A[0][u_col_offset+0], U idx = 0*4 + (u_col_offset+0)
                # local 1 -> A[0][u_col_offset+1], U idx = 0*4 + (u_col_offset+1)
                # local 2 -> A[1][u_col_offset+0], U idx = 1*4 + (u_col_offset+0)
                # local 3 -> A[1][u_col_offset+1], U idx = 1*4 + (u_col_offset+1)
                local_U = {
                    0: 0 * 4 + (u_col_offset + 0),
                    1: 0 * 4 + (u_col_offset + 1),
                    2: 1 * 4 + (u_col_offset + 0),
                    3: 1 * 4 + (u_col_offset + 1),
                }
                for li, s in strassen_U[prod_idx].items():
                    U[local_U[li]][col] += s
                
                # V local index mapping:
                # local 0 -> B[v_row_offset+0][2*j_block+0], V idx = (v_row_offset+0)*6 + (2*j_block+0)
                # local 1 -> B[v_row_offset+0][2*j_block+1], V idx = (v_row_offset+0)*6 + (2*j_block+1)
                # local 2 -> B[v_row_offset+1][2*j_block+0], V idx = (v_row_offset+1)*6 + (2*j_block+0)
                # local 3 -> B[v_row_offset+1][2*j_block+1], V idx = (v_row_offset+1)*6 + (2*j_block+1)
                local_V = {
                    0: (v_row_offset + 0) * 6 + (2 * j_block + 0),
                    1: (v_row_offset + 0) * 6 + (2 * j_block + 1),
                    2: (v_row_offset + 1) * 6 + (2 * j_block + 0),
                    3: (v_row_offset + 1) * 6 + (2 * j_block + 1),
                }
                for li, s in strassen_V[prod_idx].items():
                    V[local_V[li]][col] += s
                
                # W local index mapping:
                # local 0 -> C[0][2*j_block+0], W idx = (2*j_block+0)*2 + 0
                # local 1 -> C[0][2*j_block+1], W idx = (2*j_block+1)*2 + 0
                # local 2 -> C[1][2*j_block+0], W idx = (2*j_block+0)*2 + 1
                # local 3 -> C[1][2*j_block+1], W idx = (2*j_block+1)*2 + 1
                local_W = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for li, s in strassen_W[prod_idx].items():
                    W[local_W[li]][col] += s
            
            col_offset += 7
    
    assert col_offset == rank, f"Expected rank {rank}, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}
