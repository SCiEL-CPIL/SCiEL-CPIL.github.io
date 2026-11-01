import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix)
    # A indices: 0: a00, 1: a01, 2: a10, 3: a11
    U_coeffs = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1]   # M7: b-d
    ]
    
    # Strassen coefficients for V (B matrix)
    # B indices for block j: 
    # e: B[0, 2j] -> index 0*12 + 2j
    # f: B[0, 2j+1] -> index 0*12 + 2j+1
    # g: B[1, 2j] -> index 1*12 + 2j
    # h: B[1, 2j+1] -> index 1*12 + 2j+1
    # 
    # M1: e+h
    # M2: e
    # M3: f-h
    # M4: g-e
    # M5: h
    # M6: e+f
    # M7: g+h
    
    # Strassen coefficients for W (C matrix)
    # C indices for block j:
    # C00: k=2j, i=0 -> 2j*2 + 0 = 4j
    # C10: k=2j, i=1 -> 2j*2 + 1 = 4j+1
    # C01: k=2j+1, i=0 -> (2j+1)*2 + 0 = 4j+2
    # C11: k=2j+1, i=1 -> (2j+1)*2 + 1 = 4j+3
    #
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C00(+1) -- Wait, M2 is in C10 and C00?
    # C00 = M1 - M3 + M2 - M5
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M4 + M6 - M7
    #
    # W coeffs per M:
    # M1: C00: 1, C11: 1
    # M2: C00: 1, C10: 1
    # M3: C00: -1, C01: 1
    # M4: C10: 1, C11: -1
    # M5: C00: -1, C01: 1
    # M6: C11: 1
    # M7: C11: -1
    
    W_coeffs = [
        [1, 0, 0, 1],   # M1: C00, C11
        [1, 1, 0, 0],   # M2: C00, C10
        [-1, 0, 1, 0],  # M3: C00, C01
        [0, 1, 0, -1],  # M4: C10, C11
        [-1, 0, 1, 0],  # M5: C00, C01
        [0, 0, 0, 1],   # M6: C11
        [0, 0, 0, -1]   # M7: C11
    ]
    
    for j in range(6):
        for t in range(7):
            c = 7 * j + t
            
            # Fill U
            for i_A in range(2):
                for j_A in range(2):
                    idx = i_A * m + j_A
                    U[idx][c] = U_coeffs[t][idx]
                    
            # Fill V
            # e: 0*12 + 2j
            # f: 0*12 + 2j+1
            # g: 1*12 + 2j
            # h: 1*12 + 2j+1
            # M1: e, h
            # M2: e
            # M3: f, -h
            # M4: g, -e
            # M5: h
            # M6: e, f
            # M7: g, h
            
            v_e_idx = 0 * p + 2 * j
            v_f_idx = 0 * p + 2 * j + 1
            v_g_idx = 1 * p + 2 * j
            v_h_idx = 1 * p + 2 * j + 1
            
            if t == 0: # M1
                V[v_e_idx][c] = 1
                V[v_h_idx][c] = 1
            elif t == 1: # M2
                V[v_e_idx][c] = 1
            elif t == 2: # M3
                V[v_f_idx][c] = 1
                V[v_h_idx][c] = -1
            elif t == 3: # M4
                V[v_g_idx][c] = 1
                V[v_e_idx][c] = -1
            elif t == 4: # M5
                V[v_h_idx][c] = 1
            elif t == 5: # M6
                V[v_e_idx][c] = 1
                V[v_f_idx][c] = 1
            elif t == 6: # M7
                V[v_g_idx][c] = 1
                V[v_h_idx][c] = 1
                
            # Fill W
            # C00: 4j
            # C10: 4j+1
            # C01: 4j+2
            # C11: 4j+3
            w_c00 = 4 * j
            w_c10 = 4 * j + 1
            w_c01 = 4 * j + 2
            w_c11 = 4 * j + 3
            
            # W_coeffs[t] is [C00, C10, C01, C11]
            W[w_c00][c] = W_coeffs[t][0]
            W[w_c10][c] = W_coeffs[t][1]
            W[w_c01][c] = W_coeffs[t][2]
            W[w_c11][c] = W_coeffs[t][3]
            
    return {"U": U, "V": V, "W": W}
