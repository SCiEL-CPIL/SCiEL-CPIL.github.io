"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using 2-level Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Macro Strassen coefficients
    # alpha[k][b][c] for X_k = sum alpha A_bc
    # beta[k][c][d] for Y_k = sum beta B_cd
    # b,c,d in {0,1}
    # k in 1..7
    
    # We use 1-based indexing for k, m in the logic, 0-based in code
    # k=0 corresponds to M_1, etc.
    
    alphas = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    
    betas = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01 - B11
        [[-1, 0], [1, 0]],# M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 1], [0, 1]]  # M7: B10 + B11
    ]
    
    # Micro Strassen coefficients for A part (X entries)
    # u_coeffs[m][u][v] where u,v in {0,1}
    # m=0 is M1, etc.
    u_coeffs = [
        [[1, 0], [0, 1]], # M1: X00 + X11
        [[0, 0], [1, 1]], # M2: X10 + X11
        [[1, 0], [0, 0]], # M3: X00
        [[0, 0], [0, 1]], # M4: X11
        [[1, 1], [0, 0]], # M5: X00 + X01
        [[-1, 0], [1, 0]],# M6: X10 - X00
        [[0, 1], [0, -1]] # M7: X01 - X11
    ]
    
    # Micro Strassen coefficients for B part (Y entries)
    # v_coeffs[m][v][w] where v,w in {0,1}
    v_coeffs = [
        [[1, 0], [0, 1]], # M1: Y00 + Y11
        [[1, 0], [0, 0]], # M2: Y00
        [[0, 1], [0, -1]],# M3: Y01 - Y11
        [[0, -1], [1, 0]],# M4: Y10 - Y00
        [[0, 0], [0, 1]], # M5: Y11
        [[1, 1], [0, 0]], # M6: Y00 + Y01
        [[0, 0], [1, 1]]  # M7: Y10 + Y11
    ]
    
    # Micro Output coefficients delta[m][u][w] for Z_uw from S_m
    # Z00, Z01, Z10, Z11
    # M1: Z00+1, Z11+1
    # M2: Z10+1, Z11-1
    # M3: Z01+1, Z11+1
    # M4: Z00+1, Z10+1
    # M5: Z00-1, Z01+1
    # M6: Z11+1
    # M7: Z00+1, Z11+1
    delta_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 1]]  # M7
    ]
    
    # Macro Output coefficients gamma[k][b][d] for C_bd from Z_k
    # C00 = Z1 + Z4 - Z5 + Z7
    # C01 = Z3 + Z5
    # C10 = Z2 + Z4
    # C11 = Z1 - Z2 + Z3 + Z6
    gamma_coeffs = [
        [[1, 0], [0, 1]], # M1: C00+1, C11+1
        [[0, 0], [1, -1]],# M2: C10+1, C11-1
        [[0, 1], [0, 1]], # M3: C01+1, C11+1
        [[1, 0], [1, 0]], # M4: C00+1, C10+1
        [[-1, 1], [0, 0]],# M5: C00-1, C01+1
        [[0, 0], [0, 1]], # M6: C11+1
        [[1, 0], [0, 1]]  # M7: C00+1, C11+1
    ]
    
    # Build U, V, W
    for k in range(7):
        for m in range(7):
            r = k * 7 + m
            
            # Update U
            # U[i*4+j, r] += u_coeffs[m][u][v] * alphas[k][b][c]
            # i = 2b+u, j = 2c+v
            for b in range(2):
                for c in range(2):
                    alpha_val = alphas[k][b][c]
                    if alpha_val == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            u_val = u_coeffs[m][u][v]
                            if u_val == 0:
                                continue
                            i = 2 * b + u
                            j = 2 * c + v
                            U[i * 4 + j][r] += u_val * alpha_val
            
            # Update V
            # V[j*4+k, r] += v_coeffs[m][v][w] * betas[k][c][d]
            # j = 2c+v, k = 2d+w
            for c in range(2):
                for d in range(2):
                    beta_val = betas[k][c][d]
                    if beta_val == 0:
                        continue
                    for v in range(2):
                        for w in range(2):
                            v_val = v_coeffs[m][v][w]
                            if v_val == 0:
                                continue
                            j = 2 * c + v
                            k_idx = 2 * d + w
                            V[j * 4 + k_idx][r] += v_val * beta_val
            
            # Update W
            # W[j*4+i, r] += delta_coeffs[m][u][w] * gamma_coeffs[k][b][d]
            # i = 2b+u, j = 2d+w
            for b in range(2):
                for d in range(2):
                    gamma_val = gamma_coeffs[k][b][d]
                    if gamma_val == 0:
                        continue
                    for u in range(2):
                        for w in range(2):
                            delta_val = delta_coeffs[m][u][w]
                            if delta_val == 0:
                                continue
                            i = 2 * b + u
                            j = 2 * d + w
                            W[j * 4 + i][r] += delta_val * gamma_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
