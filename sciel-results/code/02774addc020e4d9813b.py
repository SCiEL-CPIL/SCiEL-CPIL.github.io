import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output combinations
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 terms
    # Each term is a tuple: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {row: val}
    # V_coeffs: dict {row: val}
    # W_coeffs: dict {row: val}
    
    def get_strassen_terms():
        terms = []
        # M1
        terms.append(({0: 1, 5: 1}, {0: 1, 7: 1}, {0: 1, 3: 1})) # C00:1, C11:1
        # Wait, W indices for C00, C01, C10, C11 are 0, 1, 2, 3 respectively.
        # M1 contributes to C00 (1) and C11 (1).
        # M2 contributes to C10 (1) and C11 (-1).
        # M3 contributes to C01 (1) and C11 (1).
        # M4 contributes to C00 (1) and C10 (1).
        # M5 contributes to C00 (-1) and C01 (1).
        # M6 contributes to C11 (1).
        # M7 contributes to C00 (1).
        
        # Let's re-verify W indices.
        # C is 2x2.
        # C[0,0] -> W[0]
        # C[0,1] -> W[1]
        # C[1,0] -> W[2]
        # C[1,1] -> W[3]
        
        # M1: (A00+A11)(B00+B11)
        # A00 is U[0], A11 is U[5]. B00 is V[0], B11 is V[7].
        # C00 += 1, C11 += 1.
        terms.append(({0: 1, 5: 1}, {0: 1, 7: 1}, {0: 1, 3: 1}))
        
        # M2: (A10+A11)B00
        # A10 is U[4], A11 is U[5]. B00 is V[0].
        # C10 += 1, C11 += -1.
        terms.append(({4: 1, 5: 1}, {0: 1}, {2: 1, 3: -1}))
        
        # M3: A00(B01-B11)
        # A00 is U[0]. B01 is V[1], B11 is V[7].
        # C01 += 1, C11 += 1.
        terms.append(({0: 1}, {1: 1, 7: -1}, {1: 1, 3: 1}))
        
        # M4: A11(B10-B00)
        # A11 is U[5]. B10 is V[6], B00 is V[0].
        # C00 += 1, C10 += 1.
        terms.append(({5: 1}, {6: 1, 0: -1}, {0: 1, 2: 1}))
        
        # M5: (A00+A01)B11
        # A00 is U[0], A01 is U[1]. B11 is V[7].
        # C00 += -1, C01 += 1.
        terms.append(({0: 1, 1: 1}, {7: 1}, {0: -1, 1: 1}))
        
        # M6: (A10-A00)(B00+B01)
        # A10 is U[4], A00 is U[0]. B00 is V[0], B01 is V[1].
        # C11 += 1.
        terms.append(({4: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        
        # M7: (A01-A11)(B10+B11)
        # A01 is U[1], A11 is U[5]. B10 is V[6], B11 is V[7].
        # C00 += 1.
        terms.append(({1: 1, 5: -1}, {6: 1, 7: 1}, {0: 1}))
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # 6 instances
    # Instance 0: A_L (U 0,1,4,5), B_T1 (V 0,1,6,7), C (W 0,1,2,3)
    # Instance 1: A_L (U 0,1,4,5), B_T2 (V 2,3,8,9), C (W 4,5,6,7)
    # Instance 2: A_L (U 0,1,4,5), B_T3 (V 4,5,10,11), C (W 8,9,10,11)
    # Instance 3: A_R (U 2,3,6,7), B_B1 (V 12,13,18,19), C (W 0,1,2,3)
    # Instance 4: A_R (U 2,3,6,7), B_B2 (V 14,15,20,21), C (W 4,5,6,7)
    # Instance 5: A_R (U 2,3,6,7), B_B3 (V 16,17,22,23), C (W 8,9,10,11)
    
    instances = [
        # (U_offset, V_offset, W_offset)
        # U offset is 0 for A_L, 2 for A_R (since A is 2x4, indices 0-3 are row 0, 4-7 are row 1? No.
        # U index = i*m + j.
        # A_L: i=0,1; j=0,1.
        # i=0, j=0 -> 0. i=0, j=1 -> 1.
        # i=1, j=0 -> 4. i=1, j=1 -> 5.
        # So U rows are 0, 1, 4, 5.
        # A_R: i=0,1; j=2,3.
        # i=0, j=2 -> 2. i=0, j=3 -> 3.
        # i=1, j=2 -> 6. i=1, j=3 -> 7.
        # So U rows are 2, 3, 6, 7.
        
        # The strassen_terms use local indices 0,1,4,5 for A.
        # We need to map local indices to global U indices.
        
        # V index = j*p + k.
        # B_T1: j=0,1; k=0,1.
        # j=0, k=0 -> 0. j=0, k=1 -> 1.
        # j=1, k=0 -> 6. j=1, k=1 -> 7.
        # V rows 0, 1, 6, 7.
        
        # B_T2: j=0,1; k=2,3.
        # j=0, k=2 -> 2. j=0, k=3 -> 3.
        # j=1, k=2 -> 8. j=1, k=3 -> 9.
        # V rows 2, 3, 8, 9.
        
        # B_T3: j=0,1; k=4,5.
        # j=0, k=4 -> 4. j=0, k=5 -> 5.
        # j=1, k=4 -> 10. j=1, k=5 -> 11.
        # V rows 4, 5, 10, 11.
        
        # B_B1: j=2,3; k=0,1.
        # j=2, k=0 -> 12. j=2, k=1 -> 13.
        # j=3, k=0 -> 18. j=3, k=1 -> 19.
        # V rows 12, 13, 18, 19.
        
        # B_B2: j=2,3; k=2,3.
        # j=2, k=2 -> 14. j=2, k=3 -> 15.
        # j=3, k=2 -> 20. j=3, k=3 -> 21.
        # V rows 14, 15, 20, 21.
        
        # B_B3: j=2,3; k=4,5.
        # j=2, k=4 -> 16. j=2, k=5 -> 17.
        # j=3, k=4 -> 22. j=3, k=5 -> 23.
        # V rows 16, 17, 22, 23.
        
        # W index = k*n + i.
        # C cols 0,1 -> k=0,1.
        # k=0, i=0 -> 0. k=0, i=1 -> 1.
        # k=1, i=0 -> 2. k=1, i=1 -> 3.
        # W rows 0, 1, 2, 3.
        
        # C cols 2,3 -> k=2,3.
        # k=2, i=0 -> 4. k=2, i=1 -> 5.
        # k=3, i=0 -> 6. k=3, i=1 -> 7.
        # W rows 4, 5, 6, 7.
        
        # C cols 4,5 -> k=4,5.
        # k=4, i=0 -> 8. k=4, i=1 -> 9.
        # k=5, i=0 -> 10. k=5, i=1 -> 11.
        # W rows 8, 9, 10, 11.
    ]
    
    # Define mappings from local (0,1,4,5) to global
    def map_u(local, is_right):
        if is_right:
            # Local 0,1,4,5 -> Global 2,3,6,7
            # Local 0 (A00) -> Global 2 (A02)
            # Local 1 (A01) -> Global 3 (A03)
            # Local 4 (A10) -> Global 6 (A12)
            # Local 5 (A11) -> Global 7 (A13)
            return local + 2
        else:
            return local
            
    def map_v(local, v_block):
        # v_block 0: B_T1 (0,1,6,7)
        # v_block 1: B_T2 (2,3,8,9)
        # v_block 2: B_T3 (4,5,10,11)
        # v_block 3: B_B1 (12,13,18,19)
        # v_block 4: B_B2 (14,15,20,21)
        # v_block 5: B_B3 (16,17,22,23)
        
        # Local 0,1,6,7 are the base for v_block 0.
        # For v_block b:
        # If b < 3:
        #   offset = b * 2
        #   Global = Local + offset (for rows 0,1)
        #   Wait, V index = j*p + k.
        #   For B_T1 (j=0,1; k=0,1):
        #     Local 0 (j=0,k=0) -> Global 0
        #     Local 1 (j=0,k=1) -> Global 1
        #     Local 6 (j=1,k=0) -> Global 6
        #     Local 7 (j=1,k=1) -> Global 7
        #   For B_T2 (j=0,1; k=2,3):
        #     Local 0 -> Global 2
        #     Local 1 -> Global 3
        #     Local 6 -> Global 8
        #     Local 7 -> Global 9
        #   So Global = Local + b*2.
        
        # If b >= 3:
        #   b' = b - 3 (0,1,2)
        #   Base offset for j=2 is 2*6 = 12.
        #   For B_B1 (j=2,3; k=0,1):
        #     Local 0 (j=2,k=0) -> Global 12
        #     Local 1 (j=2,k=1) -> Global 13
        #     Local 6 (j=3,k=0) -> Global 18
        #     Local 7 (j=3,k=1) -> Global 19
        #   For B_B2 (j=2,3; k=2,3):
        #     Local 0 -> Global 14
        #     Local 1 -> Global 15
        #     Local 6 -> Global 20
        #     Local 7 -> Global 21
        #   So Global = Local + 12 + b'*2.
        
        if v_block < 3:
            return local + v_block * 2
        else:
            return local + 12 + (v_block - 3) * 2

    def map_w(local, w_block):
        # w_block 0: C cols 0,1 (W 0,1,2,3)
        # w_block 1: C cols 2,3 (W 4,5,6,7)
        # w_block 2: C cols 4,5 (W 8,9,10,11)
        # Local 0,1,2,3.
        # Global = Local + w_block * 4.
        return local + w_block * 4

    # Assign columns
    col = 0
    for inst in range(6):
        is_right = (inst >= 3)
        v_block = inst % 3
        w_block = inst % 3
        
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            for local_u, val in u_coeffs.items():
                global_u = map_u(local_u, is_right)
                U[global_u][col] = val
                
            for local_v, val in v_coeffs.items():
                global_v = map_v(local_v, v_block)
                V[global_v][col] = val
                
            for local_w, val in w_coeffs.items():
                global_w = map_w(local_w, w_block)
                W[global_w][col] = val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
