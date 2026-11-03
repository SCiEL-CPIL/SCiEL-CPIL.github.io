import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x4 matrix by a 4x4 matrix.
    
    Returns:
        dict: {'U': (8, rank), 'V': (16, rank), 'W': (8, rank)}
    """
    n, m, p = 2, 4, 4
    # Target tensor T[x, y, z]
    # x = i*m + k (row of A, col of A)
    # y = k*p + j (row of B, col of B)
    # z = j*n + i (row of C, col of C)
    # T[x, y, z] = 1 if indices match multiplication logic
    
    # We use the known rank 28 solution derived from Strassen on 2x2 blocks.
    # The solution is constructed by 4 blocks of 7 Strassen products each.
    # Block 1: A1 * B1 + A2 * B3 contributes to C1 (left half of C)
    # Block 2: A1 * B2 + A2 * B4 contributes to C2 (right half of C)
    # Wait, the previous logic was:
    # C = [A1 B1 + A2 B3, A1 B2 + A2 B4]
    # Let's map the indices carefully.
    
    # A is 2x4. Rows 0,1. Cols 0..3.
    # B is 4x4. Rows 0..3. Cols 0..3.
    # C is 2x4. Rows 0,1. Cols 0..3.
    
    # x = 4*i + k. i in {0,1}, k in {0,1,2,3}.
    # y = 4*k + j. k in {0,1,2,3}, j in {0,1,2,3}.
    # z = 2*j + i. j in {0,1,2,3}, i in {0,1}.
    
    # Let's define the 4 blocks of the 28 rank solution.
    # The 28 rank solution consists of 4 sets of 7 Strassen products.
    # Actually, the standard construction for 2x4x4 with rank 28 is:
    # 1. Compute S1 = Strassen(A1, B1) (7 terms)
    # 2. Compute S2 = Strassen(A2, B3) (7 terms)
    # 3. Compute S3 = Strassen(A1, B2) (7 terms)
    # 4. Compute S4 = Strassen(A2, B4) (7 terms)
    # C_left = S1 + S2
    # C_right = S3 + S4
    
    # Indices for A1: rows 0,1; cols 0,1.
    # Indices for A2: rows 0,1; cols 2,3.
    # Indices for B1: rows 0,1; cols 0,1.
    # Indices for B2: rows 0,1; cols 2,3.
    # Indices for B3: rows 2,3; cols 0,1.
    # Indices for B4: rows 2,3; cols 2,3.
    
    # Let's build U, V, W.
    rank = 28
    U = np.zeros((8, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((8, rank), dtype=int)
    
    # Helper to add a Strassen product block
    # A_rows: indices in U for the 2x2 matrix A (4 entries)
    # B_rows: indices in V for the 2x2 matrix B (4 entries)
    # C_rows: indices in W for the 2x2 matrix C (4 entries)
    # col_offset: starting column in U, V, W
    def add_strassen_block(A_idx, B_idx, C_idx, col_off):
        # A_idx = [a00, a01, a10, a11] mapped to U rows
        # B_idx = [b00, b01, b10, b11] mapped to V rows
        # C_idx = [c00, c01, c10, c11] mapped to W rows
        
        # Strassen 7 products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b01
        # M3 = a00(b10-b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        
        # We need to map these to linear combinations of U, V, W columns.
        # Each Strassen term is a product of a linear form in A, a linear form in B, and contributes to linear forms in C.
        # Since the tensor is T[x,y,z] = sum U[x,r] V[y,r] W[z,r],
        # and we are adding a block that computes C = A B,
        # we can set the columns of U, V, W to match the Strassen factors.
        
        # Term 1: (a00+a11) * (b00+b11) * (c00+c11) ... wait.
        # The standard Strassen decomposition is:
        # C = M1 * E11 + M2 * E12 + M3 * E21 + M4 * E22 ... no, that's not right.
        # C00 = M1 + M4 - M5 + M7
        # This means the contribution to C00 from M1 is +1, from M4 is +1, etc.
        
        # Let's define the 7 columns for this block.
        # r = col_off + 0 corresponds to M1
        # U rows: a00 (+1), a11 (+1)
        # V rows: b00 (+1), b11 (+1)
        # W rows: c00 (+1), c11 (+1)  (since M1 adds to C00 and C11)
        
        U[A_idx[0]][col_off + 0] += 1
        U[A_idx[3]][col_off + 0] += 1
        V[B_idx[0]][col_off + 0] += 1
        V[B_idx[3]][col_off + 0] += 1
        W[C_idx[0]][col_off + 0] += 1
        W[C_idx[3]][col_off + 0] += 1
        
        # r = col_off + 1 corresponds to M2
        # U: a10 (+1), a11 (+1)
        # V: b01 (+1)
        # W: c01 (+1), c10 (+1)
        
        U[A_idx[2]][col_off + 1] += 1
        U[A_idx[3]][col_off + 1] += 1
        V[B_idx[1]][col_off + 1] += 1
        W[C_idx[1]][col_off + 1] += 1
        W[C_idx[2]][col_off + 1] += 1
        
        # r = col_off + 2 corresponds to M3
        # U: a00 (+1)
        # V: b10 (+1), b11 (-1)
        # W: c01 (+1), c11 (+1)
        
        U[A_idx[0]][col_off + 2] += 1
        V[B_idx[2]][col_off + 2] += 1
        V[B_idx[3]][col_off + 2] -= 1
        W[C_idx[1]][col_off + 2] += 1
        W[C_idx[3]][col_off + 2] += 1
        
        # r = col_off + 3 corresponds to M4
        # U: a11 (+1)
        # V: b00 (+1), b10 (-1)
        # W: c00 (+1), c10 (+1)
        
        U[A_idx[3]][col_off + 3] += 1
        V[B_idx[0]][col_off + 3] += 1
        V[B_idx[2]][col_off + 3] -= 1
        W[C_idx[0]][col_off + 3] += 1
        W[C_idx[2]][col_off + 3] += 1
        
        # r = col_off + 4 corresponds to M5
        # U: a00 (+1), a01 (+1)
        # V: b11 (+1)
        # W: c00 (-1), c11 (+1)
        
        U[A_idx[0]][col_off + 4] += 1
        U[A_idx[1]][col_off + 4] += 1
        V[B_idx[3]][col_off + 4] += 1
        W[C_idx[0]][col_off + 4] -= 1
        W[C_idx[3]][col_off + 4] += 1
        
        # r = col_off + 5 corresponds to M6
        # U: a10 (+1), a00 (-1)
        # V: b00 (+1), b01 (+1)
        # W: c10 (+1)
        
        U[A_idx[2]][col_off + 5] += 1
        U[A_idx[0]][col_off + 5] -= 1
        V[B_idx[0]][col_off + 5] += 1
        V[B_idx[1]][col_off + 5] += 1
        W[C_idx[2]][col_off + 5] += 1
        
        # r = col_off + 6 corresponds to M7
        # U: a01 (+1), a11 (-1)
        # V: b10 (+1), b11 (+1)
        # W: c00 (+1)
        
        U[A_idx[1]][col_off + 6] += 1
        U[A_idx[3]][col_off + 6] -= 1
        V[B_idx[2]][col_off + 6] += 1
        V[B_idx[3]][col_off + 6] += 1
        W[C_idx[0]][col_off + 6] += 1

    # Define indices for the 4 blocks
    # A1: rows 0,1 of A (x=0,1,4,5? No. x = 4*i + k).
    # A is 2x4.
    # i=0: k=0,1,2,3 -> x=0,1,2,3
    # i=1: k=0,1,2,3 -> x=4,5,6,7
    # So A1 (cols 0,1) corresponds to x=0,1 (i=0) and x=4,5 (i=1).
    # A_idx for A1: [0, 1, 4, 5] (a00, a01, a10, a11)
    
    # A2 (cols 2,3) corresponds to x=2,3 (i=0) and x=6,7 (i=1).
    # A_idx for A2: [2, 3, 6, 7]
    
    # B is 4x4.
    # y = 4*k + j.
    # B1 (rows 0,1; cols 0,1).
    # k=0, j=0,1 -> y=0,1
    # k=1, j=0,1 -> y=4,5
    # B_idx for B1: [0, 1, 4, 5] (b00, b01, b10, b11)
    
    # B2 (rows 0,1; cols 2,3).
    # k=0, j=2,3 -> y=2,3
    # k=1, j=2,3 -> y=6,7
    # B_idx for B2: [2, 3, 6, 7]
    
    # B3 (rows 2,3; cols 0,1).
    # k=2, j=0,1 -> y=8,9
    # k=3, j=0,1 -> y=12,13
    # B_idx for B3: [8, 9, 12, 13]
    
    # B4 (rows 2,3; cols 2,3).
    # k=2, j=2,3 -> y=10,11
    # k=3, j=2,3 -> y=14,15
    # B_idx for B4: [10, 11, 14, 15]
    
    # C is 2x4.
    # z = 2*j + i.
    # C1 (cols 0,1).
    # j=0, i=0,1 -> z=0,1
    # j=1, i=0,1 -> z=2,3
    # C_idx for C1: [0, 1, 2, 3] (c00, c01, c10, c11)
    
    # C2 (cols 2,3).
    # j=2, i=0,1 -> z=4,5
    # j=3, i=0,1 -> z=6,7
    # C_idx for C2: [4, 5, 6, 7]
    
    # Block 1: A1 * B1 -> C1
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3], 0)
    
    # Block 2: A2 * B3 -> C1
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3], 7)
    
    # Block 3: A1 * B2 -> C2
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7], 14)
    
    # Block 4: A2 * B4 -> C2
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7], 21)
    
    return {"U": U, "V": V, "W": W}
