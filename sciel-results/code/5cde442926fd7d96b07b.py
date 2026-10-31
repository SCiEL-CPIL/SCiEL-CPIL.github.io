import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7-term coefficients for 2x2 multiplication
    # u_t, v_t are vectors of length 4 for entries (0,0), (0,1), (1,0), (1,1)
    u_coeffs = [
        [1, 0, 0, 1], # t=0
        [0, 0, 1, 1], # t=1
        [1, 0, 0, 0], # t=2
        [0, 0, 0, 1], # t=3
        [1, 0, 1, 0], # t=4
        [-1, 1, 0, 0], # t=5
        [0, 0, 1, -1]  # t=6
    ]
    v_coeffs = [
        [1, 0, 0, 1], # t=0
        [1, 0, 0, 0], # t=1
        [0, 0, 1, -1], # t=2
        [0, 1, -1, 0], # t=3
        [0, 0, 0, 1], # t=4
        [1, 0, 1, 0], # t=5
        [0, 1, 0, 1]  # t=6
    ]
    
    # Output coefficients w_t for Z_00, Z_01, Z_10, Z_11
    # w[t][out_index] where out_index 0=00, 1=01, 2=10, 3=11
    w_coeffs = [
        [1, 0, 0, 0], # t=0 -> Z00
        [0, 0, 1, 0], # t=1 -> Z10
        [0, 1, 0, 0], # t=2 -> Z01
        [1, 0, 1, 0], # t=3 -> Z00, Z10
        [-1, 1, 0, 0], # t=4 -> Z00(-1), Z01(1)
        [0, 0, 0, 1], # t=5 -> Z11
        [1, 0, 0, 0]  # t=6 -> Z00
    ]
    # Correcting w_coeffs based on Strassen:
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    
    w_coeffs = [
        [1, 0, 0, 0], # t=0: Z00=1
        [0, 0, 1, 0], # t=1: Z10=1
        [0, 1, 0, 0], # t=2: Z01=1
        [1, 0, 1, 0], # t=3: Z00=1, Z10=1
        [-1, 1, 0, 0], # t=4: Z00=-1, Z01=1
        [0, 0, 0, 1], # t=5: Z11=1
        [1, 0, 0, 0]  # t=6: Z00=1
    ]
    # Wait, t=5 is M5. Z11 = M0 - M1 + M2 + M5. So w[5][3] = 1.
    # t=0 is M0. Z00=1, Z11=1.
    # Let's re-verify w_coeffs.
    # M0: Z00=1, Z11=1
    # M1: Z10=1, Z11=-1
    # M2: Z01=1, Z11=1
    # M3: Z00=1, Z10=1
    # M4: Z00=-1, Z01=1
    # M5: Z11=1
    # M6: Z00=1
    
    w_coeffs = [
        [1, 0, 0, 1], # t=0
        [0, 0, 1, -1], # t=1
        [0, 1, 0, 1], # t=2
        [1, 0, 1, 0], # t=3
        [-1, 1, 0, 0], # t=4
        [0, 0, 0, 1], # t=5
        [1, 0, 0, 0]  # t=6
    ]

    # Outer Strassen coefficients O_s for C_00, C_01, C_10, C_11
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    o_coeffs = [
        [1, 0, 0, 0], # s=0
        [0, 0, 1, 0], # s=1
        [0, 1, 0, 0], # s=2
        [1, 0, 1, 0], # s=3
        [-1, 1, 0, 0], # s=4
        [0, 0, 0, 1], # s=5
        [1, 0, 0, 0]  # s=6
    ]
    # Correcting o_coeffs:
    # P0: C00=1, C11=1
    # P1: C10=1, C11=-1
    # P2: C01=1, C11=1
    # P3: C00=1, C10=1
    # P4: C00=-1, C01=1
    # P5: C11=1
    # P6: C00=1
    
    o_coeffs = [
        [1, 0, 0, 1], # s=0
        [0, 0, 1, -1], # s=1
        [0, 1, 0, 1], # s=2
        [1, 0, 1, 0], # s=3
        [-1, 1, 0, 0], # s=4
        [0, 0, 0, 1], # s=5
        [1, 0, 0, 0]  # s=6
    ]

    # L_s and R_s coefficients for blocks
    # L_s[bA, cA]
    L_coeffs = [
        [[1, 0], [0, 1]], # s=0: A00, A11
        [[0, 0], [1, 1]], # s=1: A10, A11
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 0], [1, 0]], # s=4: A00, A10
        [[-1, 1], [0, 0]], # s=5: -A00, A01
        [[0, 0], [1, -1]]  # s=6: A10, -A11
    ]
    
    R_coeffs = [
        [[1, 0], [0, 1]], # s=0: B00, B11
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 0], [1, -1]], # s=2: B10, -B11
        [[-1, 1], [0, 0]], # s=3: -B00, B01
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 0], [1, 0]], # s=5: B00, B10
        [[0, 1], [0, 1]]  # s=6: B01, B11
    ]

    col = 0
    for s in range(7):
        for t in range(7):
            # Calculate U, V, W for this (s,t)
            # U[i, j] = u_t[a, b] * L_s[bA, cA] where i=2*bA+a, j=2*cA+b
            # V[j, k] = v_t[c, d] * R_s[bB, cB] where j=2*bB+c, k=2*cB+d
            # W[k, i] = o_s[rA, rB] * w_t[a_out, b_out] where i=2*rA+a_out, k=2*rB+b_out
            
            # Initialize
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            for bA in range(2):
                for cA in range(2):
                    coeff_L = L_coeffs[s][bA][cA]
                    if coeff_L == 0: continue
                    for a in range(2):
                        for b in range(2):
                            u_t_val = u_coeffs[t][a*2 + b] # Index 0:00, 1:01, 2:10, 3:11
                            i = 2 * bA + a
                            j = 2 * cA + b
                            u_vec[i * 4 + j] += coeff_L * u_t_val
            
            for bB in range(2):
                for cB in range(2):
                    coeff_R = R_coeffs[s][bB][cB]
                    if coeff_R == 0: continue
                    for c in range(2):
                        for d in range(2):
                            v_t_val = v_coeffs[t][c*2 + d]
                            j = 2 * bB + c
                            k = 2 * cB + d
                            v_vec[j * 4 + k] += coeff_R * v_t_val
            
            for rA in range(2):
                for rB in range(2):
                    coeff_O = o_coeffs[s][rA*2 + rB] # 0:00, 1:01, 2:10, 3:11
                    if coeff_O == 0: continue
                    for a_out in range(2):
                        for b_out in range(2):
                            w_t_val = w_coeffs[t][a_out*2 + b_out]
                            i = 2 * rA + a_out
                            k = 2 * rB + b_out
                            w_vec[k * 4 + i] += coeff_O * w_t_val

            for idx in range(16):
                U[idx][col] = u_vec[idx]
                V[idx][col] = v_vec[idx]
                W[idx][col] = w_vec[idx]
            
            col += 1

    return {"U": U, "V": V, "W": W}
