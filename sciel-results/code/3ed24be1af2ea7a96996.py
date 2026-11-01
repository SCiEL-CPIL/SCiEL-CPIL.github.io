import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen Decomposition
    # U2, V2, W2 are 4x7 matrices
    # Indices for 2x2:
    # U: 0:A00, 1:A01, 2:A10, 3:A11
    # V: 0:B00, 1:B01, 2:B10, 3:B11
    # W: 0:C00, 1:C10, 2:C01, 3:C11 (k*n+i)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # M1..M7
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    # Transpose to 7x4 for easier access by column t
    U2 = np.array(U2).T # 7x4
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    V2 = np.array(V2).T # 7x4
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    W2 = np.array(W2).T # 7x4
    
    # Coefficients for X_s and Y_s
    # s in 0..6
    # X_s is 2x2 matrix of block coefficients
    # Blocks: 11, 12, 21, 22
    # X_s[i][j] is a vector of 4 coefficients [c11, c12, c21, c22]
    
    # M1: (A11+A22)(B11+B22)
    X1 = np.array([
        [[1, 0, 0, 1], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 0, 0, 1]]
    ]) # This is wrong. X1 is A11+A22.
    # A11 is block 11. A22 is block 22.
    # X1[0][0] = A11[0][0] + A22[0][0].
    # So X1[0][0] corresponds to block 11 and 22.
    # X1[0][1] = A11[0][1] + A22[0][1].
    # So X1 is:
    # [ [11, 22], [11, 22] ] ? No.
    # X1 = A11 + A22.
    # X1[0][0] = A11[0][0] + A22[0][0]
    # X1[0][1] = A11[0][1] + A22[0][1]
    # X1[1][0] = A11[1][0] + A22[1][0]
    # X1[1][1] = A11[1][1] + A22[1][1]
    
    # So for each element (i,j) of the 2x2 matrix X_s,
    # the coefficients are the same for all elements?
    # Yes, because X_s is a sum of blocks.
    # X_s = c11*A11 + c12*A12 + c21*A21 + c22*A22.
    # So X_s[i][j] = c11*A11[i][j] + c12*A12[i][j] + c21*A21[i][j] + c22*A22[i][j].
    
    # Coefficients for X_s: [c11, c12, c21, c22]
    X_coeffs = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0], # M6: A21-A11
        [0, 1, 0, -1]  # M7: A12-A22
    ]
    
    Y_coeffs = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1], # M3: B12-B22
        [0, 0, 1, -1], # M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    
    # W mapping
    # M_s contributes to C blocks with signs
    # M1: C11(+1), C22(+1)
    # M2: C21(+1), C22(-1)
    # M3: C12(+1), C22(+1)
    # M4: C11(+1), C21(+1)
    # M5: C11(-1), C12(+1)
    # M6: C22(+1)
    # M7: C11(+1)
    
    # Block indices in C:
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # W2 indices: 0:C00, 1:C10, 2:C01, 3:C11 (within the 2x2 result)
    # Mapping W2 index to C block and local index
    # If M_s maps to C_block with sign S:
    # W2[0] -> C_block[0] * S
    # W2[1] -> C_block[1] * S
    # W2[2] -> C_block[2] * S
    # W2[3] -> C_block[3] * S
    
    # C_block indices:
    # C11: [0, 1, 2, 3] (local 00, 10, 01, 11) -> Global 0, 1, 2, 3?
    # Wait. C is 4x4.
    # C11 is rows 0,1 cols 0,1.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3?
    # No. Flattened C:
    # 0: (0,0)
    # 1: (0,1)
    # 2: (0,2)
    # 3: (0,3)
    # 4: (1,0)
    # 5: (1,1)
    # 6: (1,2)
    # 7: (1,3)
    # 8: (2,0)
    # ...
    # C11 block:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 4
    # (1,1) -> 5
    
    # C12 block:
    # (0,2) -> 2
    # (0,3) -> 3
    # (1,2) -> 6
    # (1,3) -> 7
    
    # C21 block:
    # (2,0) -> 8
    # (2,1) -> 9
    # (3,0) -> 12
    # (3,1) -> 13
    
    # C22 block:
    # (2,2) -> 10
    # (2,3) -> 11
    # (3,2) -> 14
    # (3,3) -> 15
    
    # W2 local indices:
    # 0: (0,0)
    # 1: (1,0)
    # 2: (0,1)
    # 3: (1,1)
    
    # So for C11:
    # W2[0] -> C[0]
    # W2[1] -> C[4]
    # W2[2] -> C[1]
    # W2[3] -> C[5]
    
    block_indices = {
        11: [0, 4, 1, 5], # Corresponding to W2 0,1,2,3
        12: [2, 6, 3, 7],
        21: [8, 12, 9, 13],
        22: [10, 14, 11, 15]
    }
    
    W_map = [
        {11: 1, 22: 1}, # M1
        {21: 1, 22: -1}, # M2
        {12: 1, 22: 1}, # M3
        {11: 1, 21: 1}, # M4
        {11: -1, 12: 1}, # M5
        {22: 1}, # M6
        {11: 1} # M7
    ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # U4
            for i in range(2):
                for j in range(2):
                    u_val = U2[t, i*2+j]
                    if u_val != 0:
                        for b in range(4):
                            c_val = X_coeffs[s][b]
                            if c_val != 0:
                                # Block b indices in A
                                # A11: 0,1,4,5
                                # A12: 2,3,6,7
                                # A21: 8,9,12,13
                                # A22: 10,11,14,15
                                # Local i,j in block
                                if b == 0: # 11
                                    idx = i*4 + j
                                elif b == 1: # 12
                                    idx = i*4 + 2 + j
                                elif b == 2: # 21
                                    idx = (i+2)*4 + j
                                else: # 22
                                    idx = (i+2)*4 + 2 + j
                                U[idx][col] += u_val * c_val
            
            # V4
            for i in range(2):
                for j in range(2):
                    v_val = V2[t, i*2+j]
                    if v_val != 0:
                        for b in range(4):
                            c_val = Y_coeffs[s][b]
                            if c_val != 0:
                                if b == 0: # 11
                                    idx = i*4 + j
                                elif b == 1: # 12
                                    idx = i*4 + 2 + j
                                elif b == 2: # 21
                                    idx = (i+2)*4 + j
                                else: # 22
                                    idx = (i+2)*4 + 2 + j
                                V[idx][col] += v_val * c_val
            
            # W4
            for w_idx in range(4):
                w_val = W2[t, w_idx]
                if w_val != 0:
                    for block, sign in W_map[s].items():
                        # block is 11, 12, 21, 22
                        # w_idx 0,1,2,3 maps to local 00,10,01,11
                        # block_indices[block][w_idx] gives global index
                        g_idx = block_indices[block][w_idx]
                        W[g_idx][col] += w_val * sign
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

