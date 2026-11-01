"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # Rank 57: 3 * 7 (Strassen for 2x2x2 blocks) + 36 (Schoolbook for 2x3x6)
    rank = 57
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coeffs (A part)
    # Indexing: A[i, j] -> U[i*m + j, col]
    # For 2x2 block, local i,j in 0..1
    # A00: i=0, j=0
    # A01: i=0, j=1
    # A10: i=1, j=0
    # A11: i=1, j=1
    
    strassen_U = [
        [1, 0, 1, 0], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 0, 1, 0], # M5: A00+A10
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    # V coeffs (B part)
    # Indexing: B[j, k] -> V[j*p + k, col]
    # For 2x2 block, local j,k in 0..1
    # B00: j=0, k=0
    # B01: j=0, k=1
    # B10: j=1, k=0
    # B11: j=1, k=1
    
    strassen_V = [
        [1, 1, 0, 0], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10-B11
        [-1, 1, 0, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    # W coeffs (C part)
    # Indexing: C[i, k] -> W[k*n + i, col]
    # C00: i=0, k=0 -> W[0]
    # C01: i=0, k=1 -> W[2]
    # C10: i=1, k=0 -> W[1]
    # C11: i=1, k=1 -> W[3]
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 0, 0, 1], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    # 3 Strassen blocks for A_L (2x2) * B_T (2x6)
    # A_L cols: 0, 1
    # B_T rows: 0, 1
    # B_T col blocks: (0,1), (2,3), (4,5)
    
    for b_col_block in range(3):
        # b_col_block 0: B cols 0,1
        # b_col_block 1: B cols 2,3
        # b_col_block 2: B cols 4,5
        
        k_offset = b_col_block * 2
        
        for s in range(7):
            # Fill U
            # A rows 0,1. Cols 0,1.
            # U index: i*m + j. m=5.
            # A00: 0*5+0=0
            # A01: 0*5+1=1
            # A10: 1*5+0=5
            # A11: 1*5+1=6
            
            U[0, col] = strassen_U[s][0]
            U[1, col] = strassen_U[s][1]
            U[5, col] = strassen_U[s][2]
            U[6, col] = strassen_U[s][3]
            
            # Fill V
            # B rows 0,1. Cols k_offset, k_offset+1.
            # V index: j*p + k. p=6.
            # B00: 0*6 + k_offset
            # B01: 0*6 + k_offset + 1
            # B10: 1*6 + k_offset
            # B11: 1*6 + k_offset + 1
            
            V[0*6 + k_offset, col] = strassen_V[s][0]
            V[0*6 + k_offset + 1, col] = strassen_V[s][1]
            V[1*6 + k_offset, col] = strassen_V[s][2]
            V[1*6 + k_offset + 1, col] = strassen_V[s][3]
            
            # Fill W
            # C rows 0,1. Cols k_offset, k_offset+1.
            # W index: k*n + i. n=2.
            # C00 (i=0, k=k_offset): k_offset*2 + 0
            # C01 (i=0, k=k_offset+1): (k_offset+1)*2 + 0
            # C10 (i=1, k=k_offset): k_offset*2 + 1
            # C11 (i=1, k=k_offset+1): (k_offset+1)*2 + 1
            
            W[k_offset*2 + 0, col] = strassen_W[s][0]
            W[(k_offset+1)*2 + 0, col] = strassen_W[s][1]
            W[k_offset*2 + 1, col] = strassen_W[s][2]
            W[(k_offset+1)*2 + 1, col] = strassen_W[s][3]
            
            col += 1
            
    # Schoolbook for A_R (2x3) * B_B (3x6)
    # A_R: rows 0,1. Cols 2,3,4.
    # B_B: rows 2,3,4. Cols 0..5.
    
    for i in range(2):
        for j in range(2, 5):
            for k in range(6):
                U[i*5 + j, col] = 1
                V[j*6 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
