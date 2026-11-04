import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Top 2x8 part: 2 * (2 * 7) = 28? No.
    # A_11 (2x2) * B_11 (2x8). B_11 has 4 2x2 blocks. Each needs 7. Total 28.
    # A_12 (2x2) * B_12 (2x8). B_12 has 4 2x2 blocks. Each needs 7. Total 28.
    # Bottom 1x8 part:
    # A_21 (1x2) * B_11 (2x8). B_11 has 4 2x2 blocks. Each M_1,2,2 needs 4. Total 16.
    # A_22 (1x2) * B_12 (2x8). B_12 has 4 2x2 blocks. Each M_1,2,2 needs 4. Total 16.
    rank = 28 + 28 + 16 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i_offset: row offset in A (0 for top 2, 2 for bottom 1? No, A_21 is 1x2)
    # j_offset: col offset in A / row offset in B
    # k_offset: col offset in B / col offset in C
    # a_rows: number of rows in A block (2 or 1)
    # b_cols: number of cols in B block (2)
    
    # We handle 2x2x2 and 1x2x2 separately.
    
    # Strassen 2x2x2
    # A local 2x2, B local 2x2, C local 2x2
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # For A_11 (rows 0,1; cols 0,1)
    # For B_11 (rows 0,1; cols 0..7)
    # We iterate 4 times for k blocks (0, 2, 4, 6)
    
    for k_block in range(4):
        k_off = k_block * 2
        
        # Block 1: A_11 (2x2) * B_11_block (2x2)
        # A rows: 0, 1. A cols: 0, 1.
        # B rows: 0, 1. B cols: k_off, k_off+1.
        # C rows: 0, 1. C cols: k_off, k_off+1.
        
        # Strassen terms
        # 1: (A00+A11)(B00+B11) -> C00, C11
        # 2: (A10+A11)B00 -> C10
        # 3: A00(B01-B11) -> C01
        # 4: A11(B10-B00) -> C00, C10
        # 5: (A00+A01)B11 -> C00, C01
        # 6: (A10-A00)(B00+B01) -> C11
        # 7: (A01-A11)(B10+B11) -> C00, C11
        
        # We will add 7 columns for this block.
        
        # Term 1
        U[0*4+0][col] = 1; U[1*4+1][col] = 1
        V[0*8+k_off][col] = 1; V[1*8+k_off+1][col] = 1
        W[k_off*3+0][col] = 1; W[(k_off+1)*3+1][col] = 1
        col += 1
        
        # Term 2
        U[1*4+0][col] = 1; U[1*4+1][col] = 1
        V[0*8+k_off][col] = 1
        W[k_off*3+1][col] = 1
        col += 1
        
        # Term 3
        U[0*4+0][col] = 1
        V[0*8+k_off+1][col] = 1; V[1*8+k_off+1][col] = -1
        W[(k_off+1)*3+0][col] = 1
        col += 1
        
        # Term 4
        U[1*4+1][col] = 1
        V[1*8+k_off][col] = 1; V[0*8+k_off][col] = -1
        W[k_off*3+0][col] = 1; W[k_off*3+1][col] = 1
        col += 1
        
        # Term 5
        U[0*4+0][col] = 1; U[0*4+1][col] = 1
        V[1*8+k_off+1][col] = 1
        W[k_off*3+0][col] = 1; W[(k_off+1)*3+0][col] = 1
        col += 1
        
        # Term 6
        U[1*4+0][col] = 1; U[0*4+0][col] = -1
        V[0*8+k_off][col] = 1; V[0*8+k_off+1][col] = 1
        W[(k_off+1)*3+1][col] = 1
        col += 1
        
        # Term 7
        U[0*4+1][col] = 1; U[1*4+1][col] = -1
        V[1*8+k_off][col] = 1; V[1*8+k_off+1][col] = 1
        W[k_off*3+0][col] = 1; W[(k_off+1)*3+1][col] = 1
        col += 1
        
        # Block 2: A_12 (2x2) * B_12_block (2x2)
        # A rows: 0, 1. A cols: 2, 3.
        # B rows: 2, 3. B cols: k_off, k_off+1.
        # C rows: 0, 1. C cols: k_off, k_off+1.
        
        # Term 1
        U[0*4+2][col] = 1; U[1*4+3][col] = 1
        V[2*8+k_off][col] = 1; V[3*8+k_off+1][col] = 1
        W[k_off*3+0][col] = 1; W[(k_off+1)*3+1][col] = 1
        col += 1
        
        # Term 2
        U[1*4+2][col] = 1; U[1*4+3][col] = 1
        V[2*8+k_off][col] = 1
        W[k_off*3+1][col] = 1
        col += 1
        
        # Term 3
        U[0*4+2][col] = 1
        V[2*8+k_off+1][col] = 1; V[3*8+k_off+1][col] = -1
        W[(k_off+1)*3+0][col] = 1
        col += 1
        
        # Term 4
        U[1*4+3][col] = 1
        V[3*8+k_off][col] = 1; V[2*8+k_off][col] = -1
        W[k_off*3+0][col] = 1; W[k_off*3+1][col] = 1
        col += 1
        
        # Term 5
        U[0*4+2][col] = 1; U[0*4+3][col] = 1
        V[3*8+k_off+1][col] = 1
        W[k_off*3+0][col] = 1; W[(k_off+1)*3+0][col] = 1
        col += 1
        
        # Term 6
        U[1*4+2][col] = 1; U[0*4+2][col] = -1
        V[2*8+k_off][col] = 1; V[2*8+k_off+1][col] = 1
        W[(k_off+1)*3+1][col] = 1
        col += 1
        
        # Term 7
        U[0*4+3][col] = 1; U[1*4+3][col] = -1
        V[3*8+k_off][col] = 1; V[3*8+k_off+1][col] = 1
        W[k_off*3+0][col] = 1; W[(k_off+1)*3+1][col] = 1
        col += 1
        
        # Block 3: A_21 (1x2) * B_11_block (2x2)
        # A row: 2. A cols: 0, 1.
        # B rows: 0, 1. B cols: k_off, k_off+1.
        # C row: 2. C cols: k_off, k_off+1.
        # M_1,2,2 Rank 4.
        # C_k = A0 B0k + A1 B1k
        # Term 1: A0 B00 -> C0
        # Term 2: A0 B01 -> C1
        # Term 3: A1 B10 -> C0
        # Term 4: A1 B11 -> C1
        
        U[2*4+0][col] = 1
        V[0*8+k_off][col] = 1
        W[k_off*3+2][col] = 1
        col += 1
        
        U[2*4+0][col] = 1
        V[0*8+k_off+1][col] = 1
        W[(k_off+1)*3+2][col] = 1
        col += 1
        
        U[2*4+1][col] = 1
        V[1*8+k_off][col] = 1
        W[k_off*3+2][col] = 1
        col += 1
        
        U[2*4+1][col] = 1
        V[1*8+k_off+1][col] = 1
        W[(k_off+1)*3+2][col] = 1
        col += 1
        
        # Block 4: A_22 (1x2) * B_12_block (2x2)
        # A row: 2. A cols: 2, 3.
        # B rows: 2, 3. B cols: k_off, k_off+1.
        # C row: 2. C cols: k_off, k_off+1.
        
        U[2*4+2][col] = 1
        V[2*8+k_off][col] = 1
        W[k_off*3+2][col] = 1
        col += 1
        
        U[2*4+2][col] = 1
        V[2*8+k_off+1][col] = 1
        W[(k_off+1)*3+2][col] = 1
        col += 1
        
        U[2*4+3][col] = 1
        V[3*8+k_off][col] = 1
        W[k_off*3+2][col] = 1
        col += 1
        
        U[2*4+3][col] = 1
        V[3*8+k_off+1][col] = 1
        W[(k_off+1)*3+2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
