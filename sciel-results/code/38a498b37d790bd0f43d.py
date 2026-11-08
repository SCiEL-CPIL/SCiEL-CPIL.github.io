"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2x2
    # Local indices:
    # U: i*2+j
    # V: j*2+k
    # W: k*2+i
    U_loc = [
        [1, 0, 0, 0, 1, 0, 0], # M1: A00, M5: A00
        [0, 0, 0, 0, 0, 0, 1], # M7: A01
        [0, 0, 0, 0, 1, 0, 0], # M5: A10
        [0, 0, 0, 0, 0, 0, -1],# M7: -A11
        [1, 0, 0, 0, 1, 0, 0], # M1: A11, M5: A11? No.
        # Let's re-verify Strassen U coefficients
    ]
    
    # Correct Strassen Coefficients:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # U (A) coefficients for M1..M7
    # A00: M1, M3, M5, M6(-)
    # A01: M7
    # A10: M2, M5
    # A11: M1, M2, M4, M6, M7(-)
    
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 0, 0, 1],  # A01
        [0, 1, 0, 0, 1, 0, 0],  # A10
        [1, 1, 0, 1, 0, 1, -1]  # A11
    ]
    
    # V (B) coefficients for M1..M7
    # B00: M1, M2, M4(-), M6
    # B01: M4, M7
    # B10: M3, M6
    # B11: M1, M3(-), M5, M7
    
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 0, 1],  # B01
        [0, 0, 1, 0, 0, 1, 0],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    # W (C) coefficients for M1..M7
    # C00: M1, M4, M5(-), M7
    # C01: M3, M5, M7
    # C10: M2, M4
    # C11: M1, M2(-), M3, M6
    
    # W index k*2+i
    # C00: k=0, i=0 -> 0
    # C01: k=0, i=1 -> 1
    # C10: k=1, i=0 -> 2
    # C11: k=1, i=1 -> 3
    
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 1],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    rank = 12 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    block_idx = 0
    for a in range(2):
        for b in range(2):
            for c in range(3):
                base_rank = block_idx * 7
                for r in range(7):
                    for u_l in range(4):
                        if U_loc[u_l][r] != 0:
                            i_l = u_l // 2
                            j_l = u_l % 2
                            i_g = 2 * a + i_l
                            j_g = 2 * b + j_l
                            u_g_idx = i_g * m + j_g
                            U[u_g_idx][base_rank + r] += U_loc[u_l][r]
                            
                    for v_l in range(4):
                        if V_loc[v_l][r] != 0:
                            j_l = v_l // 2
                            k_l = v_l % 2
                            j_g = 2 * b + j_l
                            k_g = 2 * c + k_l
                            v_g_idx = j_g * p + k_g
                            V[v_g_idx][base_rank + r] += V_loc[v_l][r]
                            
                    for w_l in range(4):
                        if W_loc[w_l][r] != 0:
                            k_l = w_l // 2
                            i_l = w_l % 2
                            k_g = 2 * c + k_l
                            i_g = 2 * a + i_l
                            w_g_idx = k_g * n + i_g
                            W[w_g_idx][base_rank + r] += W_loc[w_l][r]
                            
                block_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
