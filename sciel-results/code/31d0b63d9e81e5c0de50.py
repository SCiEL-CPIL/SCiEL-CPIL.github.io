import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    rank = 42
    
    # Initialize U, V, W as numpy arrays for easier manipulation
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)
    
    # Strassen 2x2x2 decomposition components
    # M1..M7
    U_strassen = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    V_strassen = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    W_strassen = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ])
    
    # We have 3 column blocks in B (size 2x2 each in terms of B's rows? No, B is 4x6)
    # B is 4 rows, 6 cols.
    # We split B into top 2 rows and bottom 2 rows.
    # And we split the 6 cols into 3 blocks of 2.
    
    col_offset = 0
    
    for cb in range(3): # 3 column blocks of B
        # Block 1: A_top (2x2) * B_top_block (2x2)
        # A indices: rows 0,1 of A? No, A is 2x4.
        # A is split into left 2 cols and right 2 cols.
        # So A_left is 2x2, A_right is 2x2.
        
        # Top part of B (rows 0,1)
        # U indices: A_left cols 0,1
        # V indices: B_top cols cb*2, cb*2+1
        
        for t in range(7):
            # U
            for lr in range(2):
                for lc in range(2):
                    val = U_strassen[t][lr*2+lc]
                    if val != 0:
                        # A_left index: row lr, col lc
                        U[lr*m + lc, col_offset] += val
                        
            # V
            for lr in range(2):
                for lc in range(2):
                    val = V_strassen[t][lr*2+lc]
                    if val != 0:
                        # B_top index: row lr, col cb*2 + lc
                        V[lr*p + (cb*2 + lc), col_offset] += val
                        
            # W
            for lr in range(2):
                for lc in range(2):
                    val = W_strassen[t][lr*2+lc]
                    if val != 0:
                        # C index: col cb*2+lc, row lr
                        W[(cb*2 + lc)*n + lr, col_offset] += val
                        
            col_offset += 1
            
        # Block 2: A_bottom (2x2) * B_bottom_block (2x2)
        # A_right is cols 2,3 of A.
        # B_bottom is rows 2,3 of B.
        
        for t in range(7):
            # U
            for lr in range(2):
                for lc in range(2):
                    val = U_strassen[t][lr*2+lc]
                    if val != 0:
                        # A_right index: row lr, col 2+lc
                        U[lr*m + (2 + lc), col_offset] += val
                        
            # V
            for lr in range(2):
                for lc in range(2):
                    val = V_strassen[t][lr*2+lc]
                    if val != 0:
                        # B_bottom index: row 2+lr, col cb*2 + lc
                        V[(2 + lr)*p + (cb*2 + lc), col_offset] += val
                        
            # W
            for lr in range(2):
                for lc in range(2):
                    val = W_strassen[t][lr*2+lc]
                    if val != 0:
                        # C index: col cb*2+lc, row lr
                        W[(cb*2 + lc)*n + lr, col_offset] += val
                        
            col_offset += 1

    # Convert back to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
