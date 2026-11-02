import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Structured Rank 36 Decomposition
    # We use the block decomposition derived in the thought process.
    
    # Helper to add a Strassen block
    # a_indices: list of 4 global indices for A (row-major 2x2)
    # b_indices: list of 4 global indices for B (row-major 2x2)
    # c_indices: list of 4 global indices for C (row-major 2x2)
    # offset: current column offset in U, V, W
    def add_strassen_block(U, V, W, a_idx, b_idx, c_idx, offset):
        # Strassen M1..M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Local indices 0..3 for 2x2 matrix
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1
        U[a_idx[0]][offset] += 1; U[a_idx[3]][offset] += 1
        V[b_idx[0]][offset] += 1; V[b_idx[3]][offset] += 1
        W[c_idx[0]][offset] += 1; W[c_idx[3]][offset] += 1
        W[c_idx[1]][offset] -= 1; W[c_idx[2]][offset] -= 1
        # Wait, C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's just store the coefficients in W for each M.
        # It's easier to define the 7 rank-1 terms explicitly.
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1: (A11+A22)(B11+B22)
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # C11, C22
            # M2: (A21+A22)B11
            ({2: 1, 3: 1}, {0: 1}, {2: 1}), # C21
            # M3: A11(B21-B22)
            ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}), # C12, C22
            # M4: A22(B12-B11)
            ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}), # C11, C21
            # M5: (A11+A12)B22
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # C11, C12
            # M6: (A21-A11)(B11+B12)
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C22
            # M7: (A12-A22)(B21+B22)
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 1: -1, 2: 1}) # C11, C12, C21? 
            # Check C22 for M7: 0.
            # Check C11 for M7: 1.
            # Check C12 for M7: -1? No, C12 = M3+M5. M7 is not in C12.
            # Check C21 for M7: 1? C21 = M2+M4. M7 is not in C21.
            # My W coeffs for M7 above are wrong.
            # C11 = M1 + M4 - M5 + M7 -> M7 coeff 1
            # C12 = M3 + M5 -> M7 coeff 0
            # C21 = M2 + M4 -> M7 coeff 0
            # C22 = M1 - M2 + M3 + M6 -> M7 coeff 0
            # So M7 W is {0: 1}
        ]
        
        # Correcting M7 W coeffs:
        terms[6] = ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        
        for t in terms:
            u_c, v_c, w_c = t
            for loc_a, val in u_c.items():
                U[a_idx[loc_a]][offset] += val
            for loc_b, val in v_c.items():
                V[b_idx[loc_b]][offset] += val
            for loc_c, val in w_c.items():
                W[c_idx[loc_c]][offset] += val
            offset += 1
        return offset

    # Helper to add a simple dot product block (1x2 * 2x2 -> 1x2 or similar)
    # Actually, for 1x2 * 2x2, it's just 2 dot products.
    # A is 1x2 (indices a0, a1), B is 2x2 (indices b00, b01, b10, b11)
    # C is 1x2 (indices c0, c1)
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # Rank 2.
    def add_dot_block(U, V, W, a_idx, b_idx, c_idx, offset):
        # Term 1: A0 * B_col0
        U[a_idx[0]][offset] = 1
        V[b_idx[0]][offset] = 1; V[b_idx[2]][offset] = 1
        W[c_idx[0]][offset] = 1
        offset += 1
        
        # Term 2: A1 * B_col1
        U[a_idx[1]][offset] = 1
        V[b_idx[1]][offset] = 1; V[b_idx[3]][offset] = 1
        W[c_idx[1]][offset] = 1
        offset += 1
        return offset

    # Initialize U, V, W with zeros
    # We don't know the final rank until we sum up the blocks, but we know it's 36.
    rank = 36
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    offset = 0
    
    # Block 1: A_L (2x2) * B_T (2x2) -> C_TL (2x2)
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 2, 3
    # B_T: B[0,0], B[0,1], B[1,0], B[1,1] -> 0, 1, 2, 3
    # C_TL: C[0,0], C[0,1], C[1,0], C[1,1] -> 0, 1, 2, 3
    a_idx = [0, 1, 2, 3]
    b_idx = [0, 1, 2, 3]
    c_idx = [0, 1, 2, 3]
    offset = add_strassen_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Block 2: A_R (2x2) * B_B (2x2) -> C_TL (2x2)
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> 2, 3, 4, 5
    # B_B: B[2,0], B[2,1], B[3,0], B[3,1] -> 4, 5, 8, 9
    # C_TL: C[0,0], C[0,1], C[1,0], C[1,1] -> 0, 1, 2, 3
    a_idx = [2, 3, 4, 5]
    b_idx = [4, 5, 8, 9]
    c_idx = [0, 1, 2, 3]
    offset = add_strassen_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Block 3: A_L (2x2) * B_T' (2x2) -> C_TR (2x2)
    # A_L: 0, 1, 2, 3
    # B_T': B[0,2], B[0,3], B[1,2], B[1,3] -> 2, 3, 6, 7
    # C_TR: C[0,2], C[0,3], C[1,2], C[1,3] -> 2, 3, 6, 7
    a_idx = [0, 1, 2, 3]
    b_idx = [2, 3, 6, 7]
    c_idx = [2, 3, 6, 7]
    offset = add_strassen_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Block 4: A_R (2x2) * B_B' (2x2) -> C_TR (2x2)
    # A_R: 2, 3, 4, 5
    # B_B': B[2,2], B[2,3], B[3,2], B[3,3] -> 6, 7, 10, 11
    # C_TR: 2, 3, 6, 7
    a_idx = [2, 3, 4, 5]
    b_idx = [6, 7, 10, 11]
    c_idx = [2, 3, 6, 7]
    offset = add_strassen_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Block 5: A_BL (1x2) * B_LT (2x2) -> C_BL (1x2)
    # A_BL: A[2,0], A[2,1] -> 8, 9
    # B_LT: B[0,0], B[0,1], B[1,0], B[1,1] -> 0, 1, 2, 3
    # C_BL: C[2,0], C[2,1] -> 8, 9
    a_idx = [8, 9]
    b_idx = [0, 1, 2, 3]
    c_idx = [8, 9]
    offset = add_dot_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Block 6: A_BR (1x2) * B_LB (2x2) -> C_BL (1x2)
    # A_BR: A[2,2], A[2,3] -> 10, 11
    # B_LB: B[2,0], B[2,1], B[3,0], B[3,1] -> 4, 5, 8, 9
    # C_BL: 8, 9
    a_idx = [10, 11]
    b_idx = [4, 5, 8, 9]
    c_idx = [8, 9]
    offset = add_dot_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Block 7: A_BL (1x2) * B_RT (2x2) -> C_BR (1x2)
    # A_BL: 8, 9
    # B_RT: B[0,2], B[0,3], B[1,2], B[1,3] -> 2, 3, 6, 7
    # C_BR: C[2,2], C[2,3] -> 10, 11
    a_idx = [8, 9]
    b_idx = [2, 3, 6, 7]
    c_idx = [10, 11]
    offset = add_dot_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Block 8: A_BR (1x2) * B_RB (2x2) -> C_BR (1x2)
    # A_BR: 10, 11
    # B_RB: B[2,2], B[2,3], B[3,2], B[3,3] -> 6, 7, 10, 11
    # C_BR: 10, 11
    a_idx = [10, 11]
    b_idx = [6, 7, 10, 11]
    c_idx = [10, 11]
    offset = add_dot_block(U, V, W, a_idx, b_idx, c_idx, offset)
    
    # Verify offset is 36
    if offset != 36:
        return schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

