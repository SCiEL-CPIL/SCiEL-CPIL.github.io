import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Helper to get global indices for a 2x2 block in a 4x4 matrix
    # Block 0: 00, Block 1: 01, Block 2: 10, Block 3: 11
    # Indices in flattened 4x4 (row-major)
    def get_global_indices(block):
        if block == 0: # 00
            return [0, 1, 4, 5]
        elif block == 1: # 01
            return [2, 3, 6, 7]
        elif block == 2: # 10
            return [8, 9, 12, 13]
        else: # 11
            return [10, 11, 14, 15]

    # Helper to get Strassen 2x2x2 U, V, W
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[1,0] = 1
        V[0,0] = 1; V[1,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (A10+A11)B00
        U[1,1] = 1; U[2,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # M3 = A00(B01-B11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[0,2] = 1; W[2,2] = 1
        
        # M4 = A11(B10-B00)
        U[1,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1
        
        # M5 = (A00+A01)B11
        U[0,4] = 1; U[2,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[3,4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        W[2,5] = 1; W[3,5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[2,6] = 1; U[1,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[2,6] = 1
        
        return U, V, W

    U_222, V_222, W_222 = get_strassen_222()

    # Part 1: M_{4,4,4}
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))

    def build_222_term(a_coeffs, b_coeffs, c_coeffs):
        U_t = np.zeros((16, 7))
        V_t = np.zeros((16, 7))
        W_t = np.zeros((16, 7))
        
        for col in range(7):
            for block in range(4):
                if block in a_coeffs and a_coeffs[block] != 0:
                    coeff = a_coeffs[block]
                    a_idx = get_global_indices(block)
                    for l in range(4):
                        U_t[a_idx[l], col] += coeff * U_222[l, col]
            
            for block in range(4):
                if block in b_coeffs and b_coeffs[block] != 0:
                    coeff = b_coeffs[block]
                    b_idx = get_global_indices(block)
                    for l in range(4):
                        V_t[b_idx[l], col] += coeff * V_222[l, col]
                        
            for block in range(4):
                if block in c_coeffs and c_coeffs[block] != 0:
                    coeff = c_coeffs[block]
                    c_idx = get_global_indices(block)
                    for l in range(4):
                        W_t[c_idx[l], col] += coeff * W_222[l, col]
                            
        return U_t, V_t, W_t

    terms = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {1: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {0: 1, 1: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 1: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 3: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {1: 1, 3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1, 1: 1}}
    ]
    
    for t in terms:
        U_t, V_t, W_t = build_222_term(t['a'], t['b'], t['c'])
        U_444 = np.hstack([U_444, U_t])
        V_444 = np.hstack([V_444, V_t])
        W_444 = np.hstack([W_444, W_t])

    # Part 2: M_{4,4,3}
    # Decomposition of 2x2x3 with rank 11
    def get_223_decomp():
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Strassen part (cols 0,1)
        U[:, 0:7] = U_222
        v_map = [0, 1, 3, 4]
        for i in range(4):
            V[v_map[i], 0:7] = V_222[i, 0:7]
        W[:, 0:7] = W_222
        
        # Schoolbook part (col 2)
        U[0, 7] = 1
        V[2, 7] = 1
        W[2, 7] = 1
        
        U[1, 8] = 1
        V[5, 8] = 1
        W[2, 8] = 1
        
        U[2, 9] = 1
        V[2, 9] = 1
        W[5, 9] = 1
        
        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1
        
        return U, V, W

    U_223, V_223, W_223 = get_223_decomp()

    U_443 = np.zeros((16, 44))
    V_443 = np.zeros((12, 44))
    W_443 = np.zeros((12, 44))

    def expand_223_to_443(a_block, b_rows, c_rows):
        U_exp = np.zeros((16, 11))
        V_exp = np.zeros((12, 11))
        W_exp = np.zeros((12, 11))
        
        a_idx = get_global_indices(a_block)
        v_offset = b_rows * 6
        w_offset = c_rows * 6
        
        for col in range(11):
            for l in range(4):
                if U_223[l, col] != 0:
                    U_exp[a_idx[l], col] = U_223[l, col]
            for l in range(6):
                if V_223[l, col] != 0:
                    V_exp[v_offset + l, col] = V_223[l, col]
                if W_223[l, col] != 0:
                    W_exp[w_offset + l, col] = W_223[l, col]
                    
        return U_exp, V_exp, W_exp

    # P11: A00, B_top, C_top
    U_t, V_t, W_t = expand_223_to_443(0, 0, 0)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])
    
    # P12: A01, B_bot, C_top
    U_t, V_t, W_t = expand_223_to_443(1, 1, 0)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])
    
    # P21: A10, B_top, C_bot
    U_t, V_t, W_t = expand_223_to_443(2, 0, 1)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])
    
    # P22: A11, B_bot, C_bot
    U_t, V_t, W_t = expand_223_to_443(3, 1, 1)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])

    # Combine
    U_full = np.hstack([U_444, U_443])
    
    V_full = np.zeros((28, 93))
    V_full[0:16, 0:49] = V_444
    
    # Map V_443 to V_full
    # V_443 rows 0-5 correspond to B_top (rows 0,1 of B)
    # V_443 rows 6-11 correspond to B_bot (rows 2,3 of B)
    # B is 4x7.
    # B_top indices: 4,5,6, 11,12,13
    # B_bot indices: 18,19,20, 25,26,27
    
    # Local V_443 indices 0-5 map to B_top
    # Local 0->4, 1->5, 2->6, 3->11, 4->12, 5->13
    # Local 6-11 map to B_bot
    # Local 6->18, 7->19, 8->20, 9->25, 10->26, 11->27
    
    v_map_top = [4, 5, 6, 11, 12, 13]
    v_map_bot = [18, 19, 20, 25, 26, 27]
    
    for l in range(6):
        V_full[v_map_top[l], 49:93] = V_443[l, 0:44]
    for l in range(6):
        V_full[v_map_bot[l], 49:93] = V_443[l+6, 0:44]
        
    W_full = np.zeros((28, 93))
    W_full[0:16, 0:49] = W_444
    
    # Map W_443 to W_full
    # W_443 rows 0-5 correspond to C_top (rows 0,1 of C)
    # W_443 rows 6-11 correspond to C_bot (rows 2,3 of C)
    # C is 4x7. W index k*n+i.
    # C_top indices: 16,20,24, 17,21,25
    # C_bot indices: 18,22,26, 19,23,27
    
    w_map_top = [16, 20, 24, 17, 21, 25]
    w_map_bot = [18, 22, 26, 19, 23, 27]
    
    for l in range(6):
        W_full[w_map_top[l], 49:93] = W_443[l, 0:44]
    for l in range(6):
        W_full[w_map_bot[l], 49:93] = W_443[l+6, 0:44]

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
