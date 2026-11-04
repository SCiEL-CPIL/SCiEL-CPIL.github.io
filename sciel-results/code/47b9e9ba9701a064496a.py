import numpy as np

def get_strassen_2x2x2():
    # Returns U, V, W for 2x2x2 Strassen (rank 7)
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A00(B01-B11) -> C01
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)B11 -> C01, C00
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00, C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def get_strassen_4x4x4():
    # Returns U, V, W for 4x4x4 Strassen (rank 49)
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Map 4x4 indices to 2x2 block indices
    # Block (br, bc) has local indices (lr, lc)
    # Global index = (br*2 + lr)*4 + (bc*2 + lc)
    
    col = 0
    for c in range(7):
        # S_c and T_c are sums of A and B blocks
        # We need to determine which A and B blocks are involved in M_c
        # and their signs.
        
        # M1: A11+A22, B11+B22
        # M2: A21+A22, B11
        # M3: A11, B12-B22
        # M4: A22, B21-B11
        # M5: A11+A12, B22
        # M6: A21-A11, B11+B12
        # M7: A12-A22, B21+B22
        
        if c == 0:
            a_blocks = [(0, 1), (3, 1)]
            b_blocks = [(0, 1), (3, 1)]
        elif c == 1:
            a_blocks = [(2, 1), (3, 1)]
            b_blocks = [(0, 1)]
        elif c == 2:
            a_blocks = [(0, 1)]
            b_blocks = [(1, 1), (3, -1)]
        elif c == 3:
            a_blocks = [(3, 1)]
            b_blocks = [(2, 1), (0, -1)]
        elif c == 4:
            a_blocks = [(0, 1), (1, 1)]
            b_blocks = [(3, 1)]
        elif c == 5:
            a_blocks = [(2, 1), (0, -1)]
            b_blocks = [(0, 1), (1, 1)]
        elif c == 6:
            a_blocks = [(1, 1), (3, -1)]
            b_blocks = [(2, 1), (3, 1)]
            
        # For each of the 7 inner Strassen terms (r2 in 0..6)
        for r2 in range(7):
            # The inner term r2 corresponds to U2[:, r2], V2[:, r2], W2[:, r2]
            # U2 has non-zeros at specific local indices.
            
            # For each non-zero in U2[:, r2]
            for lr_a in range(4):
                if U2[lr_a, r2] == 0: continue
                # lr_a is local index in A (0..3)
                # lr_a = lr_row * 2 + lr_col? No, flat index.
                # 0:00, 1:01, 2:10, 3:11
                lr_row = lr_a // 2
                lr_col = lr_a % 2
                
                # This local index belongs to the A blocks defined in a_blocks
                # Wait, the inner Strassen is applied to the 2x2 matrix S_c.
                # S_c is a 2x2 matrix formed by summing A blocks.
                # The entries of S_c are linear combinations of A entries.
                # S_c[lr_row, lr_col] = sum_{bid, sign} sign * A_block(bid)[lr_row, lr_col]
                
                # So for a fixed lr_a, we sum over a_blocks.
                for bid, sign in a_blocks:
                    # Global index of A_block(bid) at (lr_row, lr_col)
                    br = bid // 2
                    bc = bid % 2
                    gr = br * 2 + lr_row
                    gc = bc * 2 + lr_col
                    g_idx = gr * 4 + gc
                    U[g_idx, col] += U2[lr_a, r2] * sign
                    
            for lr_b in range(4):
                if V2[lr_b, r2] == 0: continue
                lr_row = lr_b // 2
                lr_col = lr_b % 2
                for bid, sign in b_blocks:
                    br = bid // 2
                    bc = bid % 2
                    gr = br * 2 + lr_row
                    gc = bc * 2 + lr_col
                    g_idx = gr * 4 + gc
                    V[g_idx, col] += V2[lr_b, r2] * sign
                    
            for lr_c in range(4):
                if W2[lr_c, r2] == 0: continue
                lr_row = lr_c // 2
                lr_col = lr_c % 2
                # W_c contributes to C blocks
                # c=0: C11, C22
                # c=1: C21
                # c=2: C12, C22
                # c=3: C11, C22
                # c=4: C12, C11
                # c=5: C22
                # c=6: C11, C22
                
                if c == 0:
                    c_blocks = [(0, 1), (3, 1)]
                elif c == 1:
                    c_blocks = [(2, 1)]
                elif c == 2:
                    c_blocks = [(1, 1), (3, -1)]
                elif c == 3:
                    c_blocks = [(0, 1), (3, -1)]
                elif c == 4:
                    c_blocks = [(1, 1), (0, -1)]
                elif c == 5:
                    c_blocks = [(3, 1)]
                elif c == 6:
                    c_blocks = [(0, 1), (3, 1)]
                
                for bid, sign in c_blocks:
                    br = bid // 2
                    bc = bid % 2
                    gr = br * 2 + lr_row
                    gc = bc * 2 + lr_col
                    g_idx = gr * 4 + gc
                    W[g_idx, col] += W2[lr_c, r2] * sign
                    
            col += 1
            
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    # Rank 114: 49 (Strassen 4x4x4) + 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1)
    
    U1, V1, W1 = get_strassen_4x4x4()
    U2, V2, W2 = get_strassen_4x4x4()
    
    # Schoolbook 4x4x1
    U3 = np.zeros((16, 16))
    V3 = np.zeros((36, 16))
    W3 = np.zeros((36, 16))
    for i in range(4):
        for j in range(4):
            k = 0 # local col index for B_right
            idx = i * 4 + j
            U3[idx, i*4+j] = 1
            V3[j * 9 + 8, i*4+j] = 1 # Global col 8
            W3[8 * 4 + i, i*4+j] = 1
            
    # Concatenate
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    # Adjust W indices for V2 and W2
    # V1, W1 correspond to B cols 0-3
    # V2, W2 correspond to B cols 4-7
    # V3, W3 correspond to B col 8
    
    # V2 indices: j*9 + k. k in 0..3. Need to shift to 4..7.
    # V2 was generated for 4x4x4, so k in 0..3.
    # We need to map k -> k+4.
    # V2 shape (16, 49).
    # Indices in V2 are j*4 + k? No, V2 is for 4x4 B, so shape (16, 49).
    # My get_strassen_4x4x4 returns V of shape (16, 49) where index is j*4+k.
    # But construct expects V of shape (36, rank) where index is j*9+k.
    
    # I need to convert V1, V2 from 4x4 format to 4x9 format.
    
    def convert_V(V_4x4, col_offset):
        V_4x9 = np.zeros((36, V_4x4.shape[1]))
        for j in range(4):
            for k in range(4):
                V_4x9[j * 9 + (k + col_offset), :] = V_4x4[j * 4 + k, :]
        return V_4x9
        
    def convert_W(W_4x4, col_offset):
        W_4x9 = np.zeros((36, W_4x4.shape[1]))
        for k in range(4):
            for i in range(4):
                W_4x9[(k + col_offset) * 4 + i, :] = W_4x4[k * 4 + i, :]
        return W_4x9
        
    V1_conv = convert_V(V1, 0)
    V2_conv = convert_V(V2, 4)
    W1_conv = convert_W(W1, 0)
    W2_conv = convert_W(W2, 4)
    
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1_conv, V2_conv, V3])
    W = np.hstack([W1_conv, W2_conv, W3])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
