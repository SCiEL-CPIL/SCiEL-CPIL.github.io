import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # We decompose M_{2,4,7} into two M_{2,2,7} blocks.
    # A = [A_L, A_R]
    # B = [B_T; B_B]
    # C = A_L B_T + A_R B_B
    
    # Each M_{2,2,7} is decomposed into 3 M_{2,2,2} and 1 M_{2,2,1}.
    
    # Total rank: 2 * (3*7 + 4) = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen 2x2x2 vectors
    # U, V, W are lists of 4 elements (indices 0,1,2,3 for 2x2 matrix)
    # Indices: 00->0, 01->1, 10->2, 11->3
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Define Schoolbook 2x2x1 vectors
    # A 2x2, B 2x1, C 1x2
    # B indices: 0 (j=0,k=0), 1 (j=1,k=0)
    # C indices: 0 (k=0,i=0), 1 (k=0,i=1)
    # Terms:
    # 1. A00 B00 C00
    # 2. A01 B10 C00
    # 3. A10 B00 C01
    # 4. A11 B10 C01
    SB_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    SB_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    SB_W = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    
    # Function to add a 2x2x2 block
    # a_row_offset: row offset in A (0 or 2? No, A is 2x4. Rows are 0,1. Cols are 0,1 or 2,3)
    # a_col_offset: col offset in A (0 or 2)
    # b_row_offset: row offset in B (0 or 2)
    # b_col_offset: col offset in B (0, 2, 4)
    def add_222(a_col_off, b_row_off, b_col_off):
        nonlocal col
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_idx = i * m + (j + a_col_off)
                    U[global_idx][col] += S_U[r][local_idx]
            # V
            for j in range(2):
                for k in range(2):
                    local_idx = j * 2 + k
                    global_idx = (j + b_row_off) * p + (k + b_col_off)
                    V[global_idx][col] += S_V[r][local_idx]
            # W
            for k in range(2):
                for i in range(2):
                    local_idx = k * 2 + i
                    global_idx = (k + b_col_off) * n + i
                    W[global_idx][col] += S_W[r][local_idx]
            col += 1
            
    # Function to add a 2x2x1 block
    def add_221(a_col_off, b_row_off, b_col_off):
        nonlocal col
        for r in range(4):
            # U
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_idx = i * m + (j + a_col_off)
                    U[global_idx][col] += SB_U[r][local_idx]
            # V
            for j in range(2):
                # k is 0 (only 1 col)
                local_idx = j
                global_idx = (j + b_row_off) * p + b_col_off
                V[global_idx][col] += SB_V[r][local_idx]
            # W
            # k is 0
            for i in range(2):
                local_idx = i
                global_idx = b_col_off * n + i
                W[global_idx][col] += SB_W[r][local_idx]
            col += 1
            
    # Block 1: A_L (cols 0,1) * B_T (rows 0,1)
    # B_T has 7 cols.
    # Split B_T into 3 2x2 blocks and 1 2x1 block.
    # Cols 0,1 -> 2x2
    add_222(0, 0, 0)
    # Cols 2,3 -> 2x2
    add_222(0, 0, 2)
    # Cols 4,5 -> 2x2
    add_222(0, 0, 4)
    # Col 6 -> 2x1
    add_221(0, 0, 6)
    
    # Block 2: A_R (cols 2,3) * B_B (rows 2,3)
    # B_B has 7 cols.
    # Cols 0,1 -> 2x2
    add_222(2, 2, 0)
    # Cols 2,3 -> 2x2
    add_222(2, 2, 2)
    # Cols 4,5 -> 2x2
    add_222(2, 2, 4)
    # Col 6 -> 2x1
    add_221(2, 2, 6)
    
    return {"U": U, "V": V, "W": W}
