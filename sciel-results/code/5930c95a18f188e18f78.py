import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # Rank 39 decomposition
    # 5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1)
    
    R = 39
    U = [[0.0] * R for _ in range(n * m)] # 4 x R
    V = [[0.0] * R for _ in range(m * p)] # 22 x R
    W = [[0.0] * R for _ in range(p * n)] # 22 x R
    
    # Strassen factors for 2x2x2
    # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # m0: (A00+A11)(B00+B11)
    # m1: (A10+A11)B00
    # m2: A01(B10-B11)
    # m3: A11(B01-B00)
    # m4: (A00+A01)B11
    # m5: (A10-A00)(B00+B01)
    # m6: (A01-A11)(B10+B11)
    
    # C00 = m0 - m3 + m5 + m6
    # C01 = m2 + m3
    # C10 = m1 + m4
    # C11 = m0 - m1 + m2 + m4
    
    U_strassen = [
        [1, 0, 0, 0, 1, 1, 0], # A00 in m0, m4, m5
        [0, 1, 1, 0, 1, 0, 1], # A01 in m2, m4, m6
        [0, 1, 0, 1, 0, -1, 0], # A10 in m1, m5
        [1, 1, 0, 1, 0, 0, -1]  # A11 in m0, m1, m3, m6
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00 in m0, m1, m3, m5
        [0, 0, 0, 1, 0, 1, 0],  # B01 in m3, m5
        [0, 0, 1, 0, 1, 0, 1],  # B10 in m2, m4, m6
        [1, 0, -1, 0, 1, 0, 1]  # B11 in m0, m2, m4, m6
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # C00 gets m0, m5? No, see formula above
        [0, 1, 1, 0], # C01 gets m2, m3
        [0, 0, 1, 1], # C10 gets m1, m4
        [1, -1, 0, 1] # C11 gets m0, -m1, m4?
    ]
    # Let's re-verify C11: m0 - m1 + m2 + m4.
    # W[3] should be [1, -1, 1, 1] for m0, m1, m2, m4?
    # Indices in W_strassen rows are C00, C01, C10, C11.
    # Cols are m0..m6.
    # C00: m0 - m3 + m5 + m6 -> [1, 0, 0, -1, 0, 1, 1]
    # C01: m2 + m3 -> [0, 0, 1, 1, 0, 0, 0]
    # C10: m1 + m4 -> [0, 1, 0, 0, 1, 0, 0]
    # C11: m0 - m1 + m2 + m4 -> [1, -1, 1, 0, 1, 0, 0]
    
    W_strassen = [
        [1, 0, 0, -1, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, -1, 1, 0, 1, 0, 0]
    ]
    
    col_offset = 0
    for b in range(5):
        # Block b uses B columns 2b, 2b+1
        # and C columns 2b, 2b+1
        
        for r in range(7):
            # U
            for i_u in range(4):
                U[i_u][col_offset + r] = U_strassen[i_u][r]
            
            # V
            # Local V 0 -> (0, 2b) -> idx 0*11 + 2b
            # Local V 1 -> (0, 2b+1) -> idx 0*11 + 2b+1
            # Local V 2 -> (1, 2b) -> idx 1*11 + 2b
            # Local V 3 -> (1, 2b+1) -> idx 1*11 + 2b+1
            local_v_idx = [
                0 * 11 + (2 * b),
                0 * 11 + (2 * b + 1),
                1 * 11 + (2 * b),
                1 * 11 + (2 * b + 1)
            ]
            for i_v in range(4):
                V[local_v_idx[i_v]][col_offset + r] = V_strassen[i_v][r]
            
            # W
            # Local W 0 -> (2b, 0) -> idx 2b*2 + 0
            # Local W 1 -> (2b, 1) -> idx 2b*2 + 1
            # Local W 2 -> (2b+1, 0) -> idx (2b+1)*2 + 0
            # Local W 3 -> (2b+1, 1) -> idx (2b+1)*2 + 1
            local_w_idx = [
                (2 * b) * 2 + 0,
                (2 * b) * 2 + 1,
                (2 * b + 1) * 2 + 0,
                (2 * b + 1) * 2 + 1
            ]
            for i_w in range(4):
                W[local_w_idx[i_w]][col_offset + r] = W_strassen[i_w][r]
        
        col_offset += 7
    
    # Remainder: Column 10
    # Schoolbook 2x2x1
    # Terms:
    # 1: A00 * B0,10 -> C0,10
    # 2: A01 * B1,10 -> C0,10
    # 3: A10 * B0,10 -> C1,10
    # 4: A11 * B1,10 -> C1,10
    
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,10), 1:(1,10)
    # W indices: 0:(10,0), 1:(10,1)
    
    # Term 1
    U[0][col_offset] = 1
    V[0 * 11 + 10][col_offset] = 1
    W[10 * 2 + 0][col_offset] = 1
    col_offset += 1
    
    # Term 2
    U[1][col_offset] = 1
    V[1 * 11 + 10][col_offset] = 1
    W[10 * 2 + 0][col_offset] = 1
    col_offset += 1
    
    # Term 3
    U[2][col_offset] = 1
    V[0 * 11 + 10][col_offset] = 1
    W[10 * 2 + 1][col_offset] = 1
    col_offset += 1
    
    # Term 4
    U[3][col_offset] = 1
    V[1 * 11 + 10][col_offset] = 1
    W[10 * 2 + 1][col_offset] = 1
    col_offset += 1
    
    return {"U": U, "V": V, "W": W}

