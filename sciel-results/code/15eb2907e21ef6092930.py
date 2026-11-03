"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # 1. Define 2x2x2 Strassen decomposition
    # U, V, W of shape (4, 7)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # s=1..7
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0] # C11
    ]
    
    # 2. Define 4x4x4 Strassen block coefficients
    # A_k[IJ] and B_k[IJ] for k=1..7
    # IJ: 00, 01, 10, 11
    A_k = {
        1: {0: 1, 3: 1}, # A00 + A11
        2: {2: 1, 3: 1}, # A10 + A11
        3: {0: 1},       # A00
        4: {3: 1},       # A11
        5: {0: 1, 1: 1}, # A00 + A01
        6: {2: 1, 0: -1},# A10 - A00
        7: {1: 1, 3: -1} # A01 - A11
    }
    B_k = {
        1: {0: 1, 3: 1}, # B00 + B11
        2: {0: 1},       # B00
        3: {1: 1, 3: -1},# B01 - B11
        4: {2: 1, 0: -1},# B10 - B00
        5: {3: 1},       # B11
        6: {0: 1, 1: 1}, # B00 + B01
        7: {2: 1, 3: 1}  # B10 + B11
    }
    W_k = {
        1: {0: 1, 3: 1}, # C00, C11
        2: {2: 1, 3: -1},# C10, C11
        3: {1: 1, 3: 1}, # C01, C11
        4: {0: 1, 2: 1}, # C00, C10
        5: {0: -1, 1: 1},# C00, C01
        6: {3: 1},       # C11
        7: {0: 1}        # C00
    }
    
    # 3. Construct 4x4x4 decomposition (rank 49)
    rank_444 = 49
    U_444 = [[0] * rank_444 for _ in range(16)]
    V_444 = [[0] * rank_444 for _ in range(16)]
    W_444 = [[0] * rank_444 for _ in range(16)]
    
    for k in range(1, 8):
        for s in range(7):
            t = 7 * (k - 1) + s
            # Update U_444
            for I in range(2):
                for J in range(2):
                    coeff_A = A_k[k].get(I * 2 + J, 0)
                    if coeff_A != 0:
                        for i_p in range(2):
                            for j_p in range(2):
                                i_global = 2 * I + i_p
                                j_global = 2 * J + j_p
                                u_val = U2[i_p * 2 + j_p][s]
                                U_444[i_global * 4 + j_global][t] += coeff_A * u_val
            # Update V_444
            for I in range(2):
                for J in range(2):
                    coeff_B = B_k[k].get(I * 2 + J, 0)
                    if coeff_B != 0:
                        for i_p in range(2):
                            for j_p in range(2):
                                i_global = 2 * I + i_p
                                j_global = 2 * J + j_p
                                v_val = V2[i_p * 2 + j_p][s]
                                V_444[i_global * 4 + j_global][t] += coeff_B * v_val
            # Update W_444
            for I in range(2):
                for J in range(2):
                    coeff_W = W_k[k].get(I * 2 + J, 0)
                    if coeff_W != 0:
                        for i_p in range(2):
                            for j_p in range(2):
                                i_global = 2 * I + i_p
                                j_global = 2 * J + j_p
                                w_val = W2[i_p * 2 + j_p][s]
                                W_444[i_global * 4 + j_global][t] += coeff_W * w_val
    
    # 4. Construct 4x4x1 schoolbook decomposition (rank 16)
    rank_441 = 16
    U_441 = [[0] * rank_441 for _ in range(16)]
    V_441 = [[0] * rank_441 for _ in range(20)] # m*p = 4*5 = 20
    W_441 = [[0] * rank_441 for _ in range(20)] # p*n = 5*4 = 20
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_441[i * 4 + j][r] = 1
            V_441[j * 5 + 4][r] = 1 # B index j*5 + k, k=4
            W_441[4 * 4 + i][r] = 1 # W index k*n + i, k=4
    
    # 5. Combine
    total_rank = rank_444 + rank_441
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)]
    W = [[0] * total_rank for _ in range(20)]
    
    for r in range(total_rank):
        if r < rank_444:
            for i in range(16):
                U[i][r] = U_444[i][r]
            for j in range(16):
                V[j][r] = V_444[j][r]
            for k in range(16):
                W[k][r] = W_444[k][r]
        else:
            r_441 = r - rank_444
            for i in range(16):
                U[i][r] = U_441[i][r_441]
            for j in range(20):
                V[j][r] = V_441[j][r_441]
            for k in range(20):
                W[k][r] = W_441[k][r_441]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
