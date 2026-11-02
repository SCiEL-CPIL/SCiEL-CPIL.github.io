"""Program for rank-minimal exact tensor decomposition of 3x3 * 3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x3 * 3x7 multiplication into three blocks:
    # Block 1: 3x3 * 3x3 (columns 0-2 of B)
    # Block 2: 3x3 * 3x3 (columns 3-5 of B)
    # Block 3: 3x3 * 3x1 (column 6 of B)
    
    # For each 3x3*3x3 block, we use a rank-22 decomposition:
    # - Strassen for the top-left 2x2*2x2 (7 terms)
    # - Schoolbook for column 2 of C (9 terms)  
    # - Schoolbook for row 2 of C, columns 0-1 (6 terms)
    # Total: 7 + 9 + 6 = 22 terms per 3x3 block
    
    # For the 3x3*3x1 block, we use schoolbook (9 terms)
    
    # Total rank: 22 + 22 + 9 = 53
    
    rank = 53
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Helper: add a 3x3*3x3 block with rank-22 decomposition
    # b_offset: column offset in B (0 for first block, 3 for second)
    def add_3x3_block(b_offset):
        nonlocal col
        
        # Strassen for top-left 2x2*2x2 -> C[0:2, 0:2]
        # P1 = (A00+A11)(B00+B11)
        U[0, col] = 1; U[3, col] = 1
        V[0, col] = 1; V[1, col] = 1
        W[0, col] = 1; W[3, col] = 1; W[1, col] = -1; W[4, col] = -1
        col += 1
        
        # P2 = (A10+A11)B00
        U[3, col] = 1; U[6, col] = 1
        V[0, col] = 1
        W[3, col] = 1; W[4, col] = -1
        col += 1
        
        # P3 = A00(B10-B11)
        U[0, col] = 1
        V[3, col] = 1; V[4, col] = -1
        W[1, col] = 1; W[4, col] = 1
        col += 1
        
        # P4 = A11(B01-B00)
        U[3, col] = 1
        V[1, col] = 1; V[0, col] = -1
        W[0, col] = 1; W[3, col] = 1
        col += 1
        
        # P5 = (A00+A01)B11
        U[0, col] = 1; U[1, col] = 1
        V[4, col] = 1
        W[0, col] = -1; W[1, col] = 1
        col += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[3, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[4, col] = 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[1, col] = 1; U[3, col] = -1
        V[3, col] = 1; V[4, col] = 1
        W[0, col] = 1; W[4, col] = -1
        col += 1
        
        # Schoolbook for C[:,2] (column 2 of C): 9 terms
        for i in range(3):
            for j in range(3):
                U[i*3 + j, col] = 1
                V[j*7 + (2 + b_offset), col] = 1
                W[(2 + b_offset)*3 + i, col] = 1
                col += 1
        
        # Schoolbook for C[2,:2] (row 2 of C, columns 0-1): 6 terms
        for j in range(3):
            for k in range(2):
                U[2*3 + j, col] = 1
                V[j*7 + (k + b_offset), col] = 1
                W[(k + b_offset)*3 + 2, col] = 1
                col += 1
    
    # Add first 3x3 block (B columns 0-2)
    add_3x3_block(0)
    
    # Add second 3x3 block (B columns 3-5)
    add_3x3_block(3)
    
    # Add 3x3*3x1 block (B column 6): schoolbook, 9 terms
    for i in range(3):
        for j in range(3):
            U[i*3 + j, col] = 1
            V[j*7 + 6, col] = 1
            W[6*3 + i, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
