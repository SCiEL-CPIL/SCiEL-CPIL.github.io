import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target Tensor T
    T = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def verify(U, V, W):
        R = U.shape[1]
        T_recon = np.zeros((n * m, m * p, n * p))
        for c in range(R):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        return True

    # Schoolbook Fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try Rank 37 Construction
    try:
        # Helper: Strassen 2x2x2
        def get_strassen_222():
            # Returns U, V, W of shape (4, 7)
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # m1 = (a+d)(e+h)
            U[:, 0] = [1, 0, 0, 1] # C00, C11
            V[:, 0] = [1, 0, 0, 1] # A00, A11
            W[:, 0] = [1, 0, 0, 1] # B00, B11
            
            # m2 = (c+d)e
            U[:, 1] = [0, 0, 1, 0] # C10
            V[:, 1] = [0, 0, 1, 1] # A10, A11
            W[:, 1] = [1, 0, 0, 0] # B00
            
            # m3 = a(f-h)
            U[:, 2] = [0, 1, 0, 0] # C01
            V[:, 2] = [1, 0, 0, 0] # A00
            W[:, 2] = [0, 1, 0, -1] # B01, -B11
            
            # m4 = d(g-e)
            U[:, 3] = [1, 0, 1, 0] # C00, C10
            V[:, 3] = [0, 0, 0, 1] # A11
            W[:, 3] = [0, 0, 1, -1] # B10, -B00
            
            # m5 = (a+b)h
            U[:, 4] = [1, 0, 0, -1] # C00, -C11
            V[:, 4] = [1, 1, 0, 0] # A00, A01
            W[:, 4] = [0, 0, 0, 1] # B11
            
            # m6 = (c-a)(e+f)
            U[:, 5] = [0, 0, 0, 1] # C11
            V[:, 5] = [-1, 0, 1, 0] # -A00, A10
            W[:, 5] = [1, 1, 0, 0] # B00, B01
            
            # m7 = (b-d)(g+h)
            U[:, 6] = [1, 0, 0, 1] # C00, C11
            V[:, 6] = [0, 1, 0, -1] # A01, -A11
            W[:, 6] = [0, 0, 1, 1] # B10, B11
            
            return U, V, W

        # Helper: 2x2x3 (Strassen on first 2 cols of B, Schoolbook on last col)
        def get_223():
            # A: 2x2 (4), B: 2x3 (6), C: 2x3 (6)
            # Rank 11
            U = np.zeros((6, 11))
            V = np.zeros((4, 11))
            W = np.zeros((6, 11))
            
            # First 7 terms: Strassen on B[:, 0:2]
            U_s, V_s, W_s = get_strassen_222()
            # U_s is for C 2x2. We need C 2x3.
            # C indices: 00, 01, 02, 10, 11, 12
            # Strassen C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # We map 0->0, 1->1, 2->3, 3->4
            idx_map_c = [0, 1, 3, 4]
            for r in range(7):
                for local_idx, val in enumerate(U_s[:, r]):
                    if val != 0:
                        U[idx_map_c[local_idx], r] = val
            
            # V_s is for A 2x2. No change.
            V[:, :7] = V_s
            
            # W_s is for B 2x2 (cols 0,1).
            # B indices: 00, 01, 02, 10, 11, 12
            # Strassen B indices: 00, 01, 10, 11 -> 0, 1, 3, 4
            idx_map_b = [0, 1, 3, 4]
            for r in range(7):
                for local_idx, val in enumerate(W_s[:, r]):
                    if val != 0:
                        W[idx_map_b[local_idx], r] = val
            
            # Next 4 terms: Schoolbook for B[:, 2]
            # C02 = A00*B02 + A01*B12
            # C12 = A10*B02 + A11*B12
            
            # Term 7: A00 B02 -> C02
            U[2, 7] = 1 # C02
            V[0, 7] = 1 # A00
            W[2, 7] = 1 # B02
            
            # Term 8: A01 B12 -> C02
            U[2, 8] = 1 # C02
            V[1, 8] = 1 # A01
            W[5, 8] = 1 # B12
            
            # Term 9: A10 B02 -> C12
            U[5, 9] = 1 # C12
            V[2, 9] = 1 # A10
            W[2, 9] = 1 # B02
            
            # Term 10: A11 B12 -> C12
            U[5, 10] = 1 # C12
            V[3, 10] = 1 # A11
            W[5, 10] = 1 # B12
            
            return U, V, W

        # Helper: 1x4x5 (Rank 1)
        def get_145():
            U = np.zeros((5, 1))
            V = np.zeros((4, 1))
            W = np.zeros((5, 1))
            U[0, 0] = 1
            V[:, 0] = 1
            W[:, 0] = 1
            return U, V, W

        # Assemble 3x4x5
        # A: 3x4, B: 4x5, C: 3x5
        # Split A into A_top (2x4) and A_bot (1x4)
        # Split B into B_top (2x5) and B_bot (2x5)
        
        # Part 1: A_top (2x4) * B_top (2x5)
        # Split A_top into A_L (2x2) and A_R (2x2)
        # Split B_top into B_TL (2x2) and B_TR (2x3)
        
        # Term 1.1: A_L * B_TL (2x2x2) -> Rank 7
        U1, V1, W1 = get_strassen_222()
        # Map to global indices
        # C: rows 0,1. cols 0,1.
        # A: rows 0,1. cols 0,1.
        # B: rows 0,1. cols 0,1.
        
        # Part 1.2: A_L * B_TR (2x2x3) -> Rank 11
        U2, V2, W2 = get_223()
        # C: rows 0,1. cols 2,3,4.
        # A: rows 0,1. cols 0,1.
        # B: rows 0,1. cols 2,3,4.
        
        # Part 2.1: A_R * B_BL (2x2x2) -> Rank 7
        # B_BL is rows 2,3 of B, cols 0,1.
        U3, V3, W3 = get_strassen_222()
        
        # Part 2.2: A_R * B_BR (2x2x3) -> Rank 11
        # B_BR is rows 2,3 of B, cols 2,3,4.
        U4, V4, W4 = get_223()
        
        # Part 3: A_bot (1x4) * B (4x5) -> Rank 1
        U5, V5, W5 = get_145()
        
        # Total Rank = 7 + 11 + 7 + 11 + 1 = 37
        R_total = 37
        
        U_full = np.zeros((15, R_total))
        V_full = np.zeros((12, R_total))
        W_full = np.zeros((20, R_total))
        
        # Helper to place block
        def place_block(U_block, V_block, W_block, row_offset_u, col_offset_u, 
                        row_offset_v, col_offset_v, 
                        row_offset_w, col_offset_w, 
                        col_offset_c, row_offset_b, col_offset_b,
                        U_full, V_full, W_full, start_col):
            # U_block is for C subblock
            # V_block is for A subblock
            # W_block is for B subblock
            
            # C indices: (row_offset_c + r) * 5 + (col_offset_c + c)
            # A indices: (row_offset_v + r) * 4 + (col_offset_v + c)
            # B indices: (row_offset_w + r) * 5 + (col_offset_w + c)
            
            n_c = U_block.shape[0]
            m_c = U_block.shape[1]
            
            for r in range(m_c):
                for i in range(U_block.shape[0]):
                    val_u = U_block[i, r]
                    if val_u != 0:
                        # Map local C index i to global
                        # i is 0..3 for 2x2, 0..5 for 2x3
                        # We need to know the shape of the C subblock
                        # This is getting complicated with generic mapping.
                        pass

        # Let's do it manually for clarity and speed
        
        # Block 1.1: A_L (0:2, 0:2) * B_TL (0:2, 0:2) -> C (0:2, 0:2)
        # U1 (4x7), V1 (4x7), W1 (4x7)
        # C local: 0,1,2,3 -> Global: 0,1,5,6 (rows 0,1; cols 0,1)
        # A local: 0,1,2,3 -> Global: 0,1,4,5 (rows 0,1; cols 0,1)
        # B local: 0,1,2,3 -> Global: 0,1,5,6 (rows 0,1; cols 0,1)
        
        c_map = [0, 1, 5, 6]
        a_map = [0, 1, 4, 5]
        b_map = [0, 1, 5, 6]
        
        for r in range(7):
            for i in range(4):
                if U1[i, r] != 0:
                    U_full[c_map[i], r] = U1[i, r]
                if V1[i, r] != 0:
                    V_full[a_map[i], r] = V1[i, r]
                if W1[i, r] != 0:
                    W_full[b_map[i], r] = W1[i, r]
                    
        # Block 1.2: A_L (0:2, 0:2) * B_TR (0:2, 2:5) -> C (0:2, 2:5)
        # U2 (6x11), V2 (4x11), W2 (6x11)
        # C local: 0,1,2,3,4,5 -> Global: 2,3,4, 7,8,9 (rows 0,1; cols 2,3,4)
        # A local: 0,1,2,3 -> Global: 0,1,4,5
        # B local: 0,1,2,3,4,5 -> Global: 2,3,4, 7,8,9 (rows 0,1; cols 2,3,4)
        
        c_map = [2, 3, 4, 7, 8, 9]
        a_map = [0, 1, 4, 5]
        b_map = [2, 3, 4, 7, 8, 9]
        
        start_r = 7
        for r in range(11):
            for i in range(6):
                if U2[i, r] != 0:
                    U_full[c_map[i], start_r + r] = U2[i, r]
            for i in range(4):
                if V2[i, r] != 0:
                    V_full[a_map[i], start_r + r] = V2[i, r]
            for i in range(6):
                if W2[i, r] != 0:
                    W_full[b_map[i], start_r + r] = W2[i, r]
                    
        # Block 2.1: A_R (0:2, 2:4) * B_BL (2:4, 0:2) -> C (0:2, 0:2)
        # U3 (4x7), V3 (4x7), W3 (4x7)
        # C local: 0,1,2,3 -> Global: 0,1,5,6
        # A local: 0,1,2,3 -> Global: 2,3,6,7 (rows 0,1; cols 2,3)
        # B local: 0,1,2,3 -> Global: 10,11,15,16 (rows 2,3; cols 0,1)
        
        c_map = [0, 1, 5, 6]
        a_map = [2, 3, 6, 7]
        b_map = [10, 11, 15, 16]
        
        start_r = 18
        for r in range(7):
            for i in range(4):
                if U3[i, r] != 0:
                    U_full[c_map[i], start_r + r] = U3[i, r]
                if V3[i, r] != 0:
                    V_full[a_map[i], start_r + r] = V3[i, r]
                if W3[i, r] != 0:
                    W_full[b_map[i], start_r + r] = W3[i, r]
                    
        # Block 2.2: A_R (0:2, 2:4) * B_BR (2:4, 2:5) -> C (0:2, 2:5)
        # U4 (6x11), V4 (4x11), W4 (6x11)
        # C local: 0,1,2,3,4,5 -> Global: 2,3,4, 7,8,9
        # A local: 0,1,2,3 -> Global: 2,3,6,7
        # B local: 0,1,2,3,4,5 -> Global: 12,13,14, 17,18,19
        
        c_map = [2, 3, 4, 7, 8, 9]
        a_map = [2, 3, 6, 7]
        b_map = [12, 13, 14, 17, 18, 19]
        
        start_r = 25
        for r in range(11):
            for i in range(6):
                if U4[i, r] != 0:
                    U_full[c_map[i], start_r + r] = U4[i, r]
            for i in range(4):
                if V4[i, r] != 0:
                    V_full[a_map[i], start_r + r] = V4[i, r]
            for i in range(6):
                if W4[i, r] != 0:
                    W_full[b_map[i], start_r + r] = W4[i, r]
                    
        # Block 3: A_bot (2:3, 0:4) * B (0:4, 0:5) -> C (2:3, 0:5)
        # U5 (5x1), V5 (4x1), W5 (5x1)
        # C local: 0,1,2,3,4 -> Global: 10,11,12,13,14 (row 2; cols 0..4)
        # A local: 0,1,2,3 -> Global: 8,9,10,11 (row 2; cols 0..3)
        # B local: 0,1,2,3,4 -> Global: 0,1,2,3,4 (rows 0..3? No, B is 4x5)
        # B indices: r*5+c.
        # A_bot is row 2 of A.
        # B is full.
        # C is row 2 of C.
        
        c_map = [10, 11, 12, 13, 14]
        a_map = [8, 9, 10, 11]
        b_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        start_r = 36
        for i in range(5):
            if U5[i, 0] != 0:
                U_full[c_map[i], start_r] = U5[i, 0]
        for i in range(4):
            if V5[i, 0] != 0:
                V_full[a_map[i], start_r] = V5[i, 0]
        for i in range(20):
            if W5[i % 5, 0] != 0: # W5 is 5x1, but B is 20.
                # W5 from get_145 is size 5.
                # We need W to be size 20.
                # The rank 1 term for 1x4x5 has W of size 5.
                # But here B is 4x5.
                # My get_145 assumed B is 1x5? No, 1x4x5 means A is 1x4, B is 4x5?
                # No, M_{1,4,5} means A is 1x4, B is 4x5.
                # My get_145 returned W of size 5.
                # This is incorrect for B of size 20.
                pass

        # Correction for Block 3
        # A_bot (1x4) * B (4x5).
        # C_k = sum_j A_j B_jk.
        # Tensor T_{i,j,k} with i=0 (local), j=0..3, k=0..4.
        # Rank 1 decomposition:
        # U = [1,1,1,1,1]^T (size 5)
        # V = [1,1,1,1]^T (size 4)
        # W = [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1]^T (size 20)
        
        U_full[10:15, 36] = 1
        V_full[8:12, 36] = 1
        W_full[:, 36] = 1
        
        if verify(U_full, V_full, W_full):
            best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            best_rank = 37
            
    except Exception as e:
        pass

    return best_result
