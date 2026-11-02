import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition
    # Block 1: cols 0,1 (Strassen)
    # Block 2: cols 2,3 (Strassen)
    # Block 3: col 4 (Schoolbook)
    
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen terms for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B10-B11) -> C01
    # M4: A11(B01+B11) -> C01, C00
    # M5: (A00+A10)B11 -> C10
    # M6: (A11-A10)(B01-B11) -> C10, C11
    # M7: (A00+A01)B00 -> C00, C01
    
    # U vectors (4x1)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    u1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    u3 = [1, 0, 0, 0]
    u4 = [0, 0, 0, 1]
    u5 = [1, 0, 1, 0]
    u6 = [0, 0, -1, 1]
    u7 = [1, 1, 0, 0]
    
    # Helper to add a block
    def add_strassen_block(col_offset):
        # V vectors (10x1)
        # B indices: j*5+k
        # j=0,1; k=0,1,2,3,4
        # For this block, k is col_offset, col_offset+1
        
        # M1: B00+B11 -> V[0*5+col_off] = 1, V[1*5+col_off+1] = 1
        # M2: B00 -> V[0*5+col_off] = 1
        # M3: B10-B11 -> V[1*5+col_off] = 1, V[1*5+col_off+1] = -1
        # M4: B01+B11 -> V[0*5+col_off+1] = 1, V[1*5+col_off+1] = 1
        # M5: B11 -> V[1*5+col_off+1] = 1
        # M6: B01-B11 -> V[0*5+col_off+1] = 1, V[1*5+col_off+1] = -1
        # M7: B00 -> V[0*5+col_off] = 1
        
        v1 = [0]*10; v1[0*5+col_offset] = 1; v1[1*5+col_offset+1] = 1
        v2 = [0]*10; v2[0*5+col_offset] = 1
        v3 = [0]*10; v3[1*5+col_offset] = 1; v3[1*5+col_offset+1] = -1
        v4 = [0]*10; v4[0*5+col_offset+1] = 1; v4[1*5+col_offset+1] = 1
        v5 = [0]*10; v5[1*5+col_offset+1] = 1
        v6 = [0]*10; v6[0*5+col_offset+1] = 1; v6[1*5+col_offset+1] = -1
        v7 = [0]*10; v7[0*5+col_offset] = 1
        
        # W vectors (10x1)
        # C indices: k*2+i
        # k is col_offset, col_offset+1
        # i is 0,1
        
        # M1: C00, C11 -> W[col_off*2+0]=1, W[(col_off+1)*2+1]=1
        # M2: C10 -> W[col_off*2+1]=1
        # M3: C01 -> W[(col_off+1)*2+0]=1
        # M4: C01, C00 -> W[(col_off+1)*2+0]=1, W[col_off*2+0]=1
        # M5: C10 -> W[col_off*2+1]=1
        # M6: C10, C11 -> W[col_off*2+1]=1, W[(col_off+1)*2+1]=1
        # M7: C00, C01 -> W[col_off*2+0]=1, W[(col_off+1)*2+0]=1
        
        w1 = [0]*10; w1[col_offset*2+0] = 1; w1[(col_offset+1)*2+1] = 1
        w2 = [0]*10; w2[col_offset*2+1] = 1
        w3 = [0]*10; w3[(col_offset+1)*2+0] = 1
        w4 = [0]*10; w4[(col_offset+1)*2+0] = 1; w4[col_offset*2+0] = 1
        w5 = [0]*10; w5[col_offset*2+1] = 1
        w6 = [0]*10; w6[col_offset*2+1] = 1; w6[(col_offset+1)*2+1] = 1
        w7 = [0]*10; w7[col_offset*2+0] = 1; w7[(col_offset+1)*2+0] = 1
        
        return [u1, u2, u3, u4, u5, u6, u7], [v1, v2, v3, v4, v5, v6, v7], [w1, w2, w3, w4, w5, w6, w7]
    
    # Add Block 1 (cols 0,1)
    u_bs, v_bs, w_bs = add_strassen_block(0)
    for i in range(7):
        U[:, i] = u_bs[i]
        V[:, i] = v_bs[i]
        W[:, i] = w_bs[i]
        
    # Add Block 2 (cols 2,3)
    u_bs2, v_bs2, w_bs2 = add_strassen_block(2)
    for i in range(7):
        U[:, 7+i] = u_bs2[i]
        V[:, 7+i] = v_bs2[i]
        W[:, 7+i] = w_bs2[i]
        
    # Add Block 3 (col 4)
    # M1: A00 B04 -> C04
    # M2: A01 B14 -> C04
    # M3: A10 B04 -> C14
    # M4: A11 B14 -> C14
    
    # U
    u8 = [1, 0, 0, 0]
    u9 = [0, 1, 0, 0]
    u10 = [0, 0, 1, 0]
    u11 = [0, 0, 0, 1]
    
    # V
    v8 = [0]*10; v8[0*5+4] = 1
    v9 = [0]*10; v9[1*5+4] = 1
    v10 = [0]*10; v10[0*5+4] = 1
    v11 = [0]*10; v11[1*5+4] = 1
    
    # W
    w8 = [0]*10; w8[4*2+0] = 1
    w9 = [0]*10; w9[4*2+0] = 1
    w10 = [0]*10; w10[4*2+1] = 1
    w11 = [0]*10; w11[4*2+1] = 1
    
    U[:, 14] = u8
    V[:, 14] = v8
    W[:, 14] = w8
    
    U[:, 15] = u9
    V[:, 15] = v9
    W[:, 15] = w9
    
    U[:, 16] = u10
    V[:, 16] = v10
    W[:, 16] = w10
    
    U[:, 17] = u11
    V[:, 17] = v11
    W[:, 17] = w11
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
