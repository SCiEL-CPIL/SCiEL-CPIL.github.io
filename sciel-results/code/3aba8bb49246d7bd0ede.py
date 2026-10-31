"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U2, V2, W2 are 4x7 matrices (indices 0-3 for 2x2 entries, 0-6 for terms)
    # Entry index mapping: (row, col) -> row*2 + col
    # Terms:
    # 0: (A00+A11)(B00+B11)
    # 1: (A10+A11)B00
    # 2: A00(B01-B11)
    # 3: A11(B10-B00)
    # 4: (A00+A01)B11
    # 5: (A10-A00)(B00+B01)
    # 6: (A01-A11)(B10+B11)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00 (idx 0)
        [0, 0, 1, 0, 1, 0, 1],  # A01 (idx 1)
        [0, 1, 0, 0, 0, 1, 0],  # A10 (idx 2)
        [1, 1, 0, 1, 0, 0, -1]  # A11 (idx 3)
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00 (idx 0)
        [0, 0, 1, 0, 0, 1, 0],  # B01 (idx 1)
        [0, 0, 0, 1, 0, 0, 1],  # B10 (idx 2)
        [1, 0, -1, 0, 1, 0, 1]  # B11 (idx 3)
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over outer Strassen terms T
    for T in range(7):
        # Iterate over inner Strassen terms T'
        for T_prime in range(7):
            col = T * 7 + T_prime
            
            # Calculate U coefficients
            # U[global_idx, col] = sum of products of U2 coeffs
            # Global A index I_A = 2*RA + lp, J_A = 2*CA + lq
            for RA in range(2):
                for CA in range(2):
                    u_outer = U2[RA * 2 + CA][T]
                    if u_outer == 0: continue
                    for lp in range(2):
                        for lq in range(2):
                            u_inner = U2[lp * 2 + lq][T_prime]
                            if u_inner == 0: continue
                            I_A = 2 * RA + lp
                            J_A = 2 * CA + lq
                            U[I_A * m + J_A][col] += u_outer * u_inner
                            
            # Calculate V coefficients
            # Global B index I_B = 2*RB + lr, J_B = 2*CB + ls
            for RB in range(2):
                for CB in range(2):
                    v_outer = V2[RB * 2 + CB][T]
                    if v_outer == 0: continue
                    for lr in range(2):
                        for ls in range(2):
                            v_inner = V2[lr * 2 + ls][T_prime]
                            if v_inner == 0: continue
                            I_B = 2 * RB + lr
                            J_B = 2 * CB + ls
                            V[I_B * p + J_B][col] += v_outer * v_inner
                            
            # Calculate W coefficients
            # Global C index I_C = 2*R + U, J_C = 2*S + V
            for R in range(2):
                for S in range(2):
                    w_outer = W2[R * 2 + S][T]
                    if w_outer == 0: continue
                    for U_local in range(2):
                        for V_local in range(2):
                            w_inner = W2[U_local * 2 + V_local][T_prime]
                            if w_inner == 0: continue
                            I_C = 2 * R + U_local
                            J_C = 2 * S + V_local
                            W[I_C * n + J_C][col] += w_outer * w_inner
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
