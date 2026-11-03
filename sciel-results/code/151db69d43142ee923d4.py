import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # 2x4x5 = 2x4x2 + 2x4x3
    # 2x4x2 = 2x2x2 + 2x2x2 (Rank 14)
    # 2x4x3 = 2x2x3 + 2x2x3 (Rank 22, using 11 each)
    
    # Helper to create Strassen 2x2x2 decomposition
    def strassen_222():
        # Returns U, V, W for 2x2x2
        U = [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0]
        ]
        V = [
            [1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 1]
        ]
        W = [
            [0, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 1, 0]
        ]
        return U, V, W

    # Helper to create 2x2x3 decomposition (Rank 11)
    def decomp_223():
        # 2x2x2 (Rank 7) + 2x2x1 (Rank 4)
        U_222, V_222, W_222 = strassen_222()
        
        # 2x2x1: A is 2x2, B is 2x1, C is 2x1
        # C_1 = A_11 B_1 + A_12 B_2
        # C_2 = A_21 B_1 + A_22 B_2
        # U_221 rows: A_11, A_12, A_21, A_22
        # V_221 rows: B_1, B_2
        # W_221 rows: C_1, C_2
        # Rank 4:
        # r1: A_11, B_1, C_1
        # r2: A_12, B_2, C_1
        # r3: A_21, B_1, C_2
        # r4: A_22, B_2, C_2
        U_221 = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_221 = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        W_221 = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        # Combine U, V, W
        # U: 4 rows, 7+4=11 cols
        U = [row + [0]*4 for row in U_222]
        for i in range(4):
            U[i] = U[i] + U_221[i]
            
        # V: 6 rows (2x2 for 222, 2x1 for 221 -> 4+2=6? No, 2x2x2 V is 4x7. 2x2x1 V is 2x4.)
        # Wait, 2x2x2 V has 4 rows (j=1,2; k=1,2).
        # 2x2x1 V has 2 rows (j=1,2; k=1).
        # So total V rows = 6.
        # Cols = 11.
        V = [row + [0]*4 for row in V_222]
        for i in range(4):
            V[i] = V[i] + [0]*4 # Already padded
        # Append V_221 rows
        for i in range(2):
            V.append([0]*7 + V_221[i])
            
        # W: 6 rows (2x2 for 222, 2x1 for 221 -> 4+2=6? No, 2x2x2 W is 4x7. 2x2x1 W is 2x4.)
        # Total W rows = 6.
        W = [row + [0]*4 for row in W_222]
        for i in range(4):
            W[i] = W[i] + [0]*4
        for i in range(2):
            W.append([0]*7 + W_221[i])
            
        return U, V, W

    # Now assemble 2x4x5
    # Partition p=5 into 2+3
    # Partition m=4 into 2+2
    
    # Block 1: 2x2x2 (i=1,2; j=1,2; k=1,2)
    # Block 2: 2x2x2 (i=1,2; j=3,4; k=1,2)
    # Block 3: 2x2x3 (i=1,2; j=1,2; k=3,4,5)
    # Block 4: 2x2x3 (i=1,2; j=3,4; k=3,4,5)
    
    U1, V1, W1 = strassen_222()
    U2, V2, W2 = strassen_222()
    U3, V3, W3 = decomp_223()
    U4, V4, W4 = decomp_223()
    
    # Total Rank = 7+7+11+11 = 36
    total_rank = 36
    
    U = [[0]*total_rank for _ in range(8)] # 2x4
    V = [[0]*total_rank for _ in range(20)] # 4x5
    W = [[0]*total_rank for _ in range(10)] # 5x2
    
    # Map indices
    # U rows: i*4 + j. i in 0,1. j in 0..3.
    # V rows: j*5 + k. j in 0..3. k in 0..4.
    # W rows: k*2 + i. k in 0..4. i in 0,1.
    
    # Block 1: i in 0,1. j in 0,1. k in 0,1.
    # U1 rows 0..3 correspond to i=0,1; j=0,1.
    # U1[0] -> i=0, j=0 -> U[0]
    # U1[1] -> i=0, j=1 -> U[1]
    # U1[2] -> i=1, j=0 -> U[4]
    # U1[3] -> i=1, j=1 -> U[5]
    # V1 rows 0..3 correspond to j=0,1; k=0,1.
    # V1[0] -> j=0, k=0 -> V[0]
    # V1[1] -> j=0, k=1 -> V[1]
    # V1[2] -> j=1, k=0 -> V[5]
    # V1[3] -> j=1, k=1 -> V[6]
    # W1 rows 0..3 correspond to k=0,1; i=0,1.
    # W1[0] -> k=0, i=0 -> W[0]
    # W1[1] -> k=0, i=1 -> W[1]
    # W1[2] -> k=1, i=0 -> W[2]
    # W1[3] -> k=1, i=1 -> W[3]
    
    for r in range(7):
        U[0][r] = U1[0][r]
        U[1][r] = U1[1][r]
        U[4][r] = U1[2][r]
        U[5][r] = U1[3][r]
        
        V[0][r] = V1[0][r]
        V[1][r] = V1[1][r]
        V[5][r] = V1[2][r]
        V[6][r] = V1[3][r]
        
        W[0][r] = W1[0][r]
        W[1][r] = W1[1][r]
        W[2][r] = W1[2][r]
        W[3][r] = W1[3][r]
        
    # Block 2: i in 0,1. j in 2,3. k in 0,1.
    # U2 rows 0..3 correspond to i=0,1; j=0,1 (local).
    # Local j=0 -> Global j=2. Local j=1 -> Global j=3.
    # U[0*4+2] = U[2]
    # U[0*4+3] = U[3]
    # U[1*4+2] = U[6]
    # U[1*4+3] = U[7]
    # V2 rows 0..3 correspond to j=0,1 (local); k=0,1.
    # Local j=0 -> Global j=2. Local j=1 -> Global j=3.
    # V[2*5+0] = V[10]
    # V[2*5+1] = V[11]
    # V[3*5+0] = V[15]
    # V[3*5+1] = V[16]
    # W2 same as W1.
    
    offset = 7
    for r in range(7):
        U[2][offset+r] = U2[0][r]
        U[3][offset+r] = U2[1][r]
        U[6][offset+r] = U2[2][r]
        U[7][offset+r] = U2[3][r]
        
        V[10][offset+r] = V2[0][r]
        V[11][offset+r] = V2[1][r]
        V[15][offset+r] = V2[2][r]
        V[16][offset+r] = V2[3][r]
        
        W[0][offset+r] = W2[0][r]
        W[1][offset+r] = W2[1][r]
        W[2][offset+r] = W2[2][r]
        W[3][offset+r] = W2[3][r]
        
    # Block 3: i in 0,1. j in 0,1. k in 2,3,4.
    # U3 rows 0..3 correspond to i=0,1; j=0,1.
    # U[0] += U3[0]
    # U[1] += U3[1]
    # U[4] += U3[2]
    # U[5] += U3[3]
    # V3 rows 0..5 correspond to j=0,1; k=0,1,2 (local).
    # Local k=0 -> Global k=2.
    # Local k=1 -> Global k=3.
    # Local k=2 -> Global k=4.
    # V[0*5+2] = V[2]
    # V[0*5+3] = V[3]
    # V[0*5+4] = V[4]
    # V[1*5+2] = V[7]
    # V[1*5+3] = V[8]
    # V[1*5+4] = V[9]
    # W3 rows 0..5 correspond to k=0,1,2 (local); i=0,1.
    # Local k=0 -> Global k=2.
    # W[2*2+0] = W[4]
    # W[2*2+1] = W[5]
    # W[3*2+0] = W[6]
    # W[3*2+1] = W[7]
    # W[4*2+0] = W[8]
    # W[4*2+1] = W[9]
    
    offset = 14
    for r in range(11):
        U[0][offset+r] = U3[0][r]
        U[1][offset+r] = U3[1][r]
        U[4][offset+r] = U3[2][r]
        U[5][offset+r] = U3[3][r]
        
        V[2][offset+r] = V3[0][r]
        V[3][offset+r] = V3[1][r]
        V[4][offset+r] = V3[2][r]
        V[7][offset+r] = V3[3][r]
        V[8][offset+r] = V3[4][r]
        V[9][offset+r] = V3[5][r]
        
        W[4][offset+r] = W3[0][r]
        W[5][offset+r] = W3[1][r]
        W[6][offset+r] = W3[2][r]
        W[7][offset+r] = W3[3][r]
        W[8][offset+r] = W3[4][r]
        W[9][offset+r] = W3[5][r]
        
    # Block 4: i in 0,1. j in 2,3. k in 2,3,4.
    offset = 25
    for r in range(11):
        U[2][offset+r] = U4[0][r]
        U[3][offset+r] = U4[1][r]
        U[6][offset+r] = U4[2][r]
        U[7][offset+r] = U4[3][r]
        
        V[12][offset+r] = V4[0][r]
        V[13][offset+r] = V4[1][r]
        V[14][offset+r] = V4[2][r]
        V[17][offset+r] = V4[3][r]
        V[18][offset+r] = V4[4][r]
        V[19][offset+r] = V4[5][r]
        
        W[4][offset+r] = W4[0][r]
        W[5][offset+r] = W4[1][r]
        W[6][offset+r] = W4[2][r]
        W[7][offset+r] = W4[3][r]
        W[8][offset+r] = W4[4][r]
        W[9][offset+r] = W4[5][r]
        
    return {"U": U, "V": V, "W": W}
