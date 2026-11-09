import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 61 solution
    # Block 1: 3x3x3 (cols 0-2)
    # Block 2: 3x3x3 (cols 3-5)
    # Block 3: 3x3x1 (col 6)
    
    # Helper to create a rank-1 term
    def add_term(U, V, W, u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        U[u_idx][r] += val_u
        V[v_idx][r] += val_v
        W[w_idx][r] += val_w

    # We will build the matrices for one 3x3x3 block and one 3x3x1 block
    # Then combine them.
    
    # 3x3x1 Decomposition (Rank 9)
    # w_i = sum_j a_ij v_j
    # U index i*3+j, V index j*1+0=j, W index 0*3+i=i
    def get_3x3x1_factors():
        rank = 9
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(3)] # p=1, so m*p = 3*1 = 3
        W = [[0]*rank for _ in range(3)] # p*n = 1*3 = 3
        r = 0
        for i in range(3):
            for j in range(3):
                U[i*3+j][r] = 1
                V[j][r] = 1
                W[i][r] = 1
                r += 1
        return U, V, W

    # 3x3x3 Decomposition (Rank 26)
    # Uses Strassen for 2x2 block + corrections
    def get_3x3x3_factors():
        rank = 26
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        
        r = 0
        # Strassen 2x2x2 (7 terms)
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = (a00+a11)(b00+b11)
        U[0][r] += 1; U[3][r] += 1
        V[0][r] += 1; V[4][r] += 1
        W[0][r] += 1; W[4][r] += 1; W[1][r] += 1; W[5][r] += 1 # Wait, W indices for C00, C01, C10, C11
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # W index k*3+i.
        # C00 -> k=0, i=0 -> 0
        # C01 -> k=1, i=0 -> 3
        # C10 -> k=0, i=1 -> 1
        # C11 -> k=1, i=1 -> 4
        
        # Let's use a list of (U_indices, V_indices, W_indices, coeffs)
        # To be safe, I will just add terms one by one.
        
        # Term 1: (a00+a11)(b00+b11)
        U[0][r] += 1; U[3][r] += 1
        V[0][r] += 1; V[4][r] += 1
        W[0][r] += 1 # C00
        W[4][r] += 1 # C11
        r += 1
        
        # Term 2: (a10+a11)b00
        U[3][r] += 1; U[6][r] += 1 # a10 is idx 3? No.
        # U index i*3+j.
        # a00=0, a01=1, a02=2
        # a10=3, a11=4, a12=5
        # a20=6, a21=7, a22=8
        # V index j*3+k.
        # b00=0, b01=1, b02=2
        # b10=3, b11=4, b12=5
        # b20=6, b21=7, b22=8
        
        # Term 2: (a10+a11)b00
        U[3][r] += 1; U[4][r] += 1
        V[0][r] += 1
        W[1][r] += 1 # C10
        W[4][r] -= 1 # C11
        r += 1
        
        # Term 3: a00(b10-b11)
        U[0][r] += 1
        V[3][r] += 1; V[4][r] -= 1
        W[3][r] += 1 # C01
        W[4][r] += 1 # C11
        r += 1
        
        # Term 4: a11(b01-b00)
        U[4][r] += 1
        V[1][r] += 1; V[0][r] -= 1
        W[0][r] += 1 # C00
        W[1][r] += 1 # C10
        r += 1
        
        # Term 5: (a00+a10)b11
        U[0][r] += 1; U[3][r] += 1
        V[4][r] += 1
        W[3][r] -= 1 # C01
        W[4][r] -= 1 # C11
        r += 1
        
        # Term 6: (a10-a00)(b00+b01)
        U[3][r] += 1; U[0][r] -= 1
        V[0][r] += 1; V[1][r] += 1
        W[4][r] += 1 # C11
        r += 1
        
        # Term 7: (a01-a11)(b10+b11)
        U[1][r] += 1; U[4][r] -= 1
        V[3][r] += 1; V[4][r] += 1
        W[0][r] += 1 # C00
        W[3][r] += 1 # C01
        r += 1
        
        # Now corrections for 3x3
        # 1. a_col b_row (2x2 outer product)
        # a_col = [a02, a12], b_row = [b20, b21]
        # C00 += a02 b20
        # C01 += a02 b21
        # C10 += a12 b20
        # C11 += a12 b21
        # Term 8: a02 b20
        U[2][r] += 1; V[6][r] += 1
        W[0][r] += 1
        r += 1
        # Term 9: a02 b21
        U[2][r] += 1; V[7][r] += 1
        W[3][r] += 1
        r += 1
        # Term 10: a12 b20
        U[5][r] += 1; V[6][r] += 1
        W[1][r] += 1
        r += 1
        # Term 11: a12 b21
        U[5][r] += 1; V[7][r] += 1
        W[4][r] += 1
        r += 1
        
        # 2. A_2x2 b_col (2x2x1 mat-vec)
        # b_col = [b02, b12]
        # C02 = a00 b02 + a01 b12
        # C12 = a10 b02 + a11 b12
        # Term 12: a00 b02
        U[0][r] += 1; V[2][r] += 1
        W[6][r] += 1 # C02 -> k=2, i=0 -> 6
        r += 1
        # Term 13: a01 b12
        U[1][r] += 1; V[5][r] += 1
        W[6][r] += 1
        r += 1
        # Term 14: a10 b02
        U[3][r] += 1; V[2][r] += 1
        W[7][r] += 1 # C12 -> k=2, i=1 -> 7
        r += 1
        # Term 15: a11 b12
        U[4][r] += 1; V[5][r] += 1
        W[7][r] += 1
        r += 1
        
        # 3. a_col b22 (vector * scalar)
        # C02 += a02 b22
        # C12 += a12 b22
        # Term 16: a02 b22
        U[2][r] += 1; V[8][r] += 1
        W[6][r] += 1
        r += 1
        # Term 17: a12 b22
        U[5][r] += 1; V[8][r] += 1
        W[7][r] += 1
        r += 1
        
        # 4. a_row B_2x2 (1x2x2 vec-mat)
        # a_row = [a20, a21]
        # C20 = a20 b00 + a21 b10
        # C21 = a20 b01 + a21 b11
        # Term 18: a20 b00
        U[6][r] += 1; V[0][r] += 1
        W[1][r] += 1 # C20 -> k=0, i=2 -> 2? No.
        # W index k*n+i. n=3.
        # C20: k=0, i=2 -> 0*3+2 = 2.
        W[2][r] += 1
        r += 1
        # Term 19: a21 b10
        U[7][r] += 1; V[3][r] += 1
        W[2][r] += 1
        r += 1
        # Term 20: a20 b01
        U[6][r] += 1; V[1][r] += 1
        W[5][r] += 1 # C21 -> k=1, i=2 -> 1*3+2 = 5
        r += 1
        # Term 21: a21 b11
        U[7][r] += 1; V[4][r] += 1
        W[5][r] += 1
        r += 1
        
        # 5. a22 b_row (scalar * vector)
        # C20 += a22 b20
        # C21 += a22 b21
        # Term 22: a22 b20
        U[8][r] += 1; V[6][r] += 1
        W[2][r] += 1
        r += 1
        # Term 23: a22 b21
        U[8][r] += 1; V[7][r] += 1
        W[5][r] += 1
        r += 1
        
        # 6. C22 = a20 b02 + a21 b12 + a22 b22
        # Term 24: a20 b02
        U[6][r] += 1; V[2][r] += 1
        W[8][r] += 1 # C22 -> k=2, i=2 -> 2*3+2 = 8
        r += 1
        # Term 25: a21 b12
        U[7][r] += 1; V[5][r] += 1
        W[8][r] += 1
        r += 1
        # Term 26: a22 b22
        U[8][r] += 1; V[8][r] += 1
        W[8][r] += 1
        r += 1
        
        return U, V, W

    # Assemble 3x3x7
    rank = 26 + 26 + 9
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(21)] # 3*7
    W = [[0]*rank for _ in range(21)] # 7*3
    
    # Block 1: Cols 0-2
    U1, V1, W1 = get_3x3x3_factors()
    for i in range(9):
        for r in range(26):
            U[i][r] = U1[i][r]
            V[i*7 + (r%3)][r] = V1[i][r] # V index j*7+k. In 3x3x3, V index is j*3+k.
            # Mapping: j*3+k -> j*7+k
            # V1 index is j*3+k.
            # We need to map V1[i] to V[j*7+k].
            # V1 is 9x26. Index 0..8.
            # j = i//3, k = i%3.
            # V index in full: j*7+k.
            
    # Let's do the mapping carefully.
    def embed_block(U_block, V_block, W_block, start_rank, col_offset):
        # U_block: 9xR
        # V_block: 9xR (indices j*3+k)
        # W_block: 9xR (indices k*3+i)
        # col_offset: 0, 3, 6
        R = U_block[0].__len__()
        for i in range(9):
            for r in range(R):
                U[i][start_rank + r] = U_block[i][r]
                
                # V mapping
                v_idx_block = i
                j = v_idx_block // 3
                k = v_idx_block % 3
                v_idx_full = j * 7 + (k + col_offset)
                V[v_idx_full][start_rank + r] = V_block[i][r]
                
                # W mapping
                w_idx_block = i
                k = w_idx_block // 3
                i_val = w_idx_block % 3
                w_idx_full = (k + col_offset) * 3 + i_val
                W[w_idx_full][start_rank + r] = W_block[i][r]

    embed_block(U1, V1, W1, 0, 0)
    
    # Block 2: Cols 3-5
    U2, V2, W2 = get_3x3x3_factors()
    embed_block(U2, V2, W2, 26, 3)
    
    # Block 3: Col 6
    U3, V3, W3 = get_3x3x1_factors()
    # U3: 9x9
    # V3: 3x9 (indices j)
    # W3: 3x9 (indices i)
    for i in range(9):
        for r in range(9):
            U[i][26 + 26 + r] = U3[i][r]
            
            # V mapping
            v_idx_block = i
            j = v_idx_block
            v_idx_full = j * 7 + 6
            V[v_idx_full][26 + 26 + r] = V3[i][r]
            
            # W mapping
            w_idx_block = i
            i_val = w_idx_block
            w_idx_full = 6 * 3 + i_val
            W[w_idx_full][26 + 26 + r] = W3[i][r]
            
    return {"U": U, "V": V, "W": W}

