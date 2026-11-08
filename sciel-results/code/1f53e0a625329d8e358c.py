import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 * 7 = 56
    rank = 56
    
    # U: (n*m, rank) = (8, 56)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank) = (32, 56)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank) = (16, 56)
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Order: X00, X01, X10, X11
    # U_s[i, j0] corresponds to X_{i, j0}
    # V_s[j0, k0] corresponds to Y_{j0, k0}
    # W_s[i, k0] corresponds to Z_{i, k0}
    
    # s=0: M1
    U_0 = [1, 0, 0, 1] # X00, X01, X10, X11 -> X00 + X11
    V_0 = [1, 0, 0, 1] # Y00, Y01, Y10, Y11 -> Y00 + Y11
    W_0 = [1, 0, 0, 1] # Z00, Z01, Z10, Z11 -> Z00 + Z11
    
    # s=1: M2
    U_1 = [0, 1, 0, 1] # X01 + X11
    V_1 = [1, 0, 0, 0] # Y00
    W_1 = [0, 1, 1, 0] # Z01 + Z10
    
    # s=2: M3
    U_2 = [1, 0, 0, 0] # X00
    V_2 = [0, 0, 1, -1] # Y10 - Y11
    W_2 = [0, 1, 1, 0] # Z01 + Z10
    
    # s=3: M4
    U_3 = [0, 0, 0, 1] # X11
    V_3 = [-1, 1, 0, 0] # Y01 - Y00
    W_3 = [1, 0, 0, 1] # Z00 + Z11
    
    # s=4: M5
    U_4 = [1, 1, 0, 0] # X00 + X01
    V_4 = [0, 0, 0, 1] # Y11
    W_4 = [-1, 1, 0, 0] # -Z00 + Z01
    
    # s=5: M6
    U_5 = [-1, 0, 1, 0] # X10 - X00
    V_5 = [1, 1, 0, 0] # Y00 + Y01
    W_5 = [0, -1, 1, 0] # -Z01 + Z10
    
    # s=6: M7
    U_6 = [0, 0, -1, 1] # X11 - X10
    V_6 = [0, 0, 1, 1] # Y10 + Y11
    W_6 = [1, 0, 0, 1] # Z00 + Z11
    
    U_coeffs = [U_0, U_1, U_2, U_3, U_4, U_5, U_6]
    V_coeffs = [V_0, V_1, V_2, V_3, V_4, V_5, V_6]
    W_coeffs = [W_0, W_1, W_2, W_3, W_4, W_5, W_6]
    
    # 8 blocks
    for b in range(8):
        j1 = b // 4
        k1 = b % 4
        
        for s in range(7):
            c = 7 * b + s
            
            u_vec = U_coeffs[s]
            v_vec = V_coeffs[s]
            w_vec = W_coeffs[s]
            
            # Fill U
            # u_vec index: i*2 + j0
            for i in range(2):
                for j0 in range(2):
                    val = u_vec[i * 2 + j0]
                    if val != 0:
                        j_global = 2 * j1 + j0
                        row_u = i * m + j_global
                        U[row_u][c] = val
                        
            # Fill V
            # v_vec index: j0*2 + k0
            for j0 in range(2):
                for k0 in range(2):
                    val = v_vec[j0 * 2 + k0]
                    if val != 0:
                        j_global = 2 * j1 + j0
                        k_global = 2 * k1 + k0
                        row_v = j_global * p + k_global
                        V[row_v][c] = val
                        
            # Fill W
            # w_vec index: i*2 + k0
            for i in range(2):
                for k0 in range(2):
                    val = w_vec[i * 2 + k0]
                    if val != 0:
                        k_global = 2 * k1 + k0
                        row_w = k_global * n + i
                        W[row_w][c] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
