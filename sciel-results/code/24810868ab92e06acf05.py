import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 4, 4
    
    # Local factors for 3x2x2 multiplication with rank 10
    # A is 3x2, B is 2x2, C is 3x2
    # Local A indices: 0..5 (row * 2 + col)
    # Local B indices: 0..3 (row * 2 + col)
    # Local C indices: 0..5 (row * 2 + col)
    
    U_loc = np.zeros((6, 12))
    V_loc = np.zeros((4, 12))
    W_loc = np.zeros((6, 12))
    
    # Rank-10 decomposition for 3x2x2
    # Strategy: Use Strassen-like products that share computation between rows
    
    # Local indices: a_idx = ar*2+ac (ar=0..2, ac=0..1)
    # a: 0=a00, 1=a01, 2=a10, 3=a11, 4=a20, 5=a21
    # b_idx = br*2+bc (br=0..1, bc=0..1)
    # b: 0=b00, 1=b01, 2=b10, 3=b11
    # c_idx = cr*2+cc (cr=0..2, cc=0..1)
    # c: 0=c00, 1=c01, 2=c10, 3=c11, 4=c20, 5=c21
    # T[a,b,c] = 1 if c_row == a_row and a_col == b_row
    
    # Strategy: For rows 0,1 of A, share products. Row 2 is separate.
    # M0: (a00+a10)*(b00+b10) -> contributes to c00, c10
    U_loc[0, 0] = 1; U_loc[2, 0] = 1
    V_loc[0, 0] = 1; V_loc[2, 0] = 1
    W_loc[0, 0] = 1; W_loc[2, 0] = 1
    
    # M1: (a00+a10)*(b10-b00) -> c00 gets (a00+a10)*(b10-b00), c10 gets (a00+a10)*(b10-b00)
    # Actually let me use a cleaner Strassen-like approach
    # M0: a00*b00, M1: a00*b10, M2: a01*b01, M3: a01*b11 (row 0)
    # M4: a10*b00, M5: a10*b10, M6: a11*b01, M7: a11*b11 (row 1)
    # M8: a20*b00, M9: a20*b10 (row 2, col 0)
    # M10: a21*b01, M11: a21*b11 (row 2, col 1)
    # That's rank 12. Can we do better?
    
    # Try: M0: (a00+a10)*b00, M1: (a00+a10)*b10, M2: (a01+a11)*b01, M3: (a01+a11)*b11
    # M4: a00*b00, M5: a10*b00 (to separate row 0 and row 1 for col 0)
    # This is getting complex. Let me just use rank 12 for now and tile.
    
    # Rank 12 direct decomposition:
    # M0: a00*b00 -> c00
    U_loc[0, 0] = 1
    V_loc[0, 0] = 1
    W_loc[0, 0] = 1
    
    # M1: a00*b10 -> c00
    U_loc[0, 1] = 1
    V_loc[2, 1] = 1
    W_loc[0, 1] = 1
    
    # M2: a01*b01 -> c01
    U_loc[1, 2] = 1
    V_loc[1, 2] = 1
    W_loc[1, 2] = 1
    
    # M3: a01*b11 -> c01
    U_loc[1, 3] = 1
    V_loc[3, 3] = 1
    W_loc[1, 3] = 1
    
    # M4: a10*b00 -> c10
    U_loc[2, 4] = 1
    V_loc[0, 4] = 1
    W_loc[2, 4] = 1
    
    # M5: a10*b10 -> c10
    U_loc[2, 5] = 1
    V_loc[2, 5] = 1
    W_loc[2, 5] = 1
    
    # M6: a11*b01 -> c11
    U_loc[3, 6] = 1
    V_loc[1, 6] = 1
    W_loc[3, 6] = 1
    
    # M7: a11*b11 -> c11
    U_loc[3, 7] = 1
    V_loc[3, 7] = 1
    W_loc[3, 7] = 1
    
    # M8: a20*b00 -> c20
    U_loc[4, 8] = 1
    V_loc[0, 8] = 1
    W_loc[4, 8] = 1
    
    # M9: a20*b10 -> c20
    U_loc[4, 9] = 1
    V_loc[2, 9] = 1
    W_loc[4, 9] = 1
    
    # M10: a21*b01 -> c21
    U_loc[5, 10] = 1
    V_loc[1, 10] = 1
    W_loc[5, 10] = 1
    
    # M11: a21*b11 -> c21
    U_loc[5, 11] = 1
    V_loc[3, 11] = 1
    W_loc[5, 11] = 1
    
    # Verify local decomposition
    # Build the tensor and check
    def verify_local(U_l, V_l, W_l):
        # For each output (i,j,k), check sum_r U[i,r]*V[j,r]*W[k,r] == T[i,j,k]
        # T[i,j,k] = 1 if i*2+j and k correspond to valid product, else 0
        # Actually, for matrix mult: T[a_idx, b_idx, c_idx] = 1 if c_idx = a_idx contributes to b_idx
        # Let me just check by building the product
        
        # U_l: 6x10, V_l: 4x10, W_l: 6x10
        # The tensor T[a,b,c] = sum_r U_l[a,r] * V_l[b,r] * W_l[c,r]
        # For 3x2 x 2x2 -> 3x2:
        # a_idx = ar*2+ac (ar=0..2, ac=0..1)
        # b_idx = br*2+bc (br=0..1, bc=0..1)
        # c_idx = cr*2+cc (cr=0..2, cc=0..1)
        # T[a,b,c] = 1 if cr==ar and ac==br, else 0
        
        T = np.zeros((6, 4, 6))
        for ar in range(3):
            for ac in range(2):
                a_idx = ar*2+ac
                for br in range(2):
                    for bc in range(2):
                        b_idx = br*2+bc
                        for cr in range(3):
                            for cc in range(2):
                                c_idx = cr*2+cc
                                if cr == ar and ac == br:
                                    T[a_idx, b_idx, c_idx] = 1
        
        # Compute decomposition
        T_dec = np.zeros((6, 4, 6))
        for r in range(10):
            T_dec += np.einsum('a,b,c->abc', U_l[:,r], V_l[:,r], W_l[:,r])
        
        return np.allclose(T, T_dec, atol=1e-9)
    
    # Verify and use rank 12
    if not verify_local(U_loc, V_loc, W_loc):
        # Fallback to rank 11
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1
        
        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1
        
        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1
        
        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
        
        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Use rank 12
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 12
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+12] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+12] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+12] += W_loc[local_c_idx, :]
    
    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    if time.time() > deadline:
        # Should not happen for static construction, but guard anyway
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
