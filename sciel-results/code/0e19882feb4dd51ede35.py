import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    
    # 1. Strassen for M_2,2,2
    # A indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # B indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # C indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    
    strassen_U = [
        [1, 1, 0, 0, 1, 0, 0], # a11
        [0, 0, 0, 0, 1, 1, -1], # a12 (Note: P5, P6, P7 coeffs)
        [0, 1, 1, 1, 0, -1, 0], # a21
        [1, 0, 0, 1, 0, 0, -1]  # a22
    ]
    # Wait, let's verify Strassen coeffs carefully.
    # P1 = (a11+a22)(b11+b22) -> U: 1,0,0,1 ; V: 1,0,0,1
    # P2 = (a21+a22)b11 -> U: 0,0,1,1 ; V: 1,0,0,0
    # P3 = a11(b21-b22) -> U: 1,0,0,0 ; V: 0,0,1,-1
    # P4 = a22(b12-b11) -> U: 0,0,0,1 ; V: -1,1,0,0
    # P5 = (a11+a12)b22 -> U: 1,1,0,0 ; V: 0,0,0,1
    # P6 = (a21-a11)(b11+b12) -> U: -1,0,1,0 ; V: 1,1,0,0
    # P7 = (a12-a22)(b12+b21) -> U: 0,1,0,-1 ; V: 0,1,1,0
    
    # C reconstruction:
    # c11 = P1 - P4 - P5 + P7
    # c12 = P3 + P5
    # c21 = P2 + P4
    # c22 = P1 - P2 - P3 + P6
    
    strassen_W = [
        [1, 0, 0, 0], # c11 from P1
        [0, 1, 0, 0], # c12 from P2? No.
        # Let's map W[k*n+i, col]
        # c11 is index 0. c12 is 1. c21 is 2. c22 is 3.
        # W[0, 0] = 1 (P1)
        # W[0, 3] = -1 (P4)
        # W[0, 4] = -1 (P5)
        # W[0, 6] = 1 (P7)
        # W[1, 2] = 1 (P3)
        # W[1, 4] = 1 (P5)
        # W[2, 1] = 1 (P2)
        # W[2, 3] = 1 (P4)
        # W[3, 0] = 1 (P1)
        # W[3, 1] = -1 (P2)
        # W[3, 2] = -1 (P3)
        # W[3, 5] = 1 (P6)
        
        [1, 0, 0, 0, 0, 0, 1], # c11 (0)
        [0, 0, 1, 0, 1, 0, 0], # c12 (1)
        [0, 1, 0, 1, 0, 0, 0], # c21 (2)
        [1, -1, -1, 0, 0, 1, 0] # c22 (3)
    ]
    
    # 2. Schoolbook for M_1,2,2 (Row 3)
    # A indices: 4:(3,1), 5:(3,2)
    # B indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # C indices: 4:(3,1), 5:(3,2)
    
    # c31 = a31 b11 + a32 b21
    # c32 = a31 b12 + a32 b22
    
    # Terms 7,8,9,10 (0-indexed 7..10)
    # T7: a31 b11 -> c31
    # T8: a32 b21 -> c31
    # T9: a31 b12 -> c32
    # T10: a32 b22 -> c32
    
    # U (6x4)
    sb_U = [
        [1, 0, 1, 0], # a31
        [0, 1, 0, 1]  # a32
    ]
    # V (4x4)
    sb_V = [
        [1, 0, 0, 0], # b11
        [0, 0, 1, 0], # b21
        [0, 1, 0, 0], # b12
        [0, 0, 0, 1]  # b22
    ]
    # W (2x4) -> maps to c31, c32
    # c31 index 4, c32 index 5
    sb_W = [
        [1, 1, 0, 0], # c31
        [0, 0, 1, 1]  # c32
    ]
    
    # Combine M_3,2,2
    # U: 6x11
    U_322 = [[0]*11 for _ in range(6)]
    V_322 = [[0]*11 for _ in range(4)]
    W_322 = [[0]*11 for _ in range(6)]
    
    # Fill Strassen (cols 0-6)
    for i in range(4):
        for c in range(7):
            U_322[i][c] = strassen_U[i][c]
            V_322[i][c] = strassen_V[i][c] # Need strassen_V
    # Need strassen_V
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # b11
        [0, 0, 0, 1, 0, 1, 1], # b12
        [0, 0, 1, 0, 0, 0, 1], # b21
        [1, 0, -1, 0, 1, 0, 0]  # b22
    ]
    
    for i in range(4):
        for c in range(7):
            U_322[i][c] = strassen_U[i][c]
            V_322[i][c] = strassen_V[i][c]
            W_322[i][c] = strassen_W[i][c] # Wait, strassen_W is 4x7. W_322 is 6x11.
            # strassen_W[i] corresponds to c_i.
            # In W_322, row i is c_i.
    
    # Fill Schoolbook (cols 7-10)
    for i in range(2):
        for c in range(4):
            U_322[4+i][7+c] = sb_U[i][c]
            V_322[i][7+c] = sb_V[i][c]
            W_322[4+i][7+c] = sb_W[i][c]
            
    # 3. Extend to M_3,2,8
    # B is 2x8. Split into 4 blocks of 2x2.
    # Block 0: B cols 0,1. B indices 0,1,2,3.
    # Block 1: B cols 2,3. B indices 4,5,6,7.
    # Block 2: B cols 4,5. B indices 8,9,10,11.
    # Block 3: B cols 6,7. B indices 12,13,14,15.
    
    # C is 3x8.
    # Block 0: C cols 0,1. C indices 0,1,2,3,4,5?
    # C index formula: k*n + i. n=3.
    # Block 0 (k=0,1): C indices 0..5.
    # Block 1 (k=2,3): C indices 6..11.
    # Block 2 (k=4,5): C indices 12..17.
    # Block 3 (k=6,7): C indices 18..23.
    
    # U is same for all blocks (A is 3x2).
    # V changes per block.
    # W changes per block.
    
    U_328 = [[0]*44 for _ in range(6)]
    V_328 = [[0]*44 for _ in range(16)]
    W_328 = [[0]*44 for _ in range(24)]
    
    for b in range(4):
        offset = b * 11
        # U
        for i in range(6):
            for c in range(11):
                U_328[i][offset+c] = U_322[i][c]
        
        # V
        # B block b uses B indices 4b .. 4b+3
        for i in range(4):
            for c in range(11):
                V_328[4*b + i][offset+c] = V_322[i][c]
                
        # W
        # C block b uses C indices 6b .. 6b+5
        for i in range(6):
            for c in range(11):
                W_328[6*b + i][offset+c] = W_322[i][c]
                
    # 4. Extend to M_3,4,8
    # A is 3x4. Split into 2 blocks of 3x2.
    # Block 0: A cols 0,1. A indices 0,1,2,3,4,5.
    # Block 1: A cols 2,3. A indices 6,7,8,9,10,11.
    
    # B is 4x8.
    # Block 0 uses B rows 0,1. B indices 0..15.
    # Block 1 uses B rows 2,3. B indices 16..31.
    
    # C is 3x8. Same for both blocks.
    
    U_348 = [[0]*88 for _ in range(12)]
    V_348 = [[0]*88 for _ in range(32)]
    W_348 = [[0]*88 for _ in range(24)]
    
    for a in range(2):
        offset = a * 44
        # U
        # A block a uses A indices 6a .. 6a+5
        for i in range(6):
            for c in range(44):
                U_348[6*a + i][offset+c] = U_328[i][c]
                
        # V
        # B block a uses B indices 16a .. 16a+15
        for i in range(16):
            for c in range(44):
                V_348[16*a + i][offset+c] = V_328[i][c]
                
        # W
        # C is same
        for i in range(24):
            for c in range(44):
                W_348[i][offset+c] = W_328[i][c]
                
    return {"U": U_348, "V": V_348, "W": W_348}
