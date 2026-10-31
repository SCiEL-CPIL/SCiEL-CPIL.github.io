import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # Safety margin, construction is fast
    
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 (Micro level)
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Products P0..P6
    # U_m[t][a][b] is coeff of x_ab in P_t
    # V_m[t][c][d] is coeff of y_cd in P_t
    
    U_m = [[[0]*2 for _ in range(2)] for _ in range(7)]
    V_m = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # P0 = (x00+x11)(y00+y11)
    U_m[0][0][0] = 1; U_m[0][1][1] = 1
    V_m[0][0][0] = 1; V_m[0][1][1] = 1
    
    # P1 = (x10+x11)y00
    U_m[1][1][0] = 1; U_m[1][1][1] = 1
    V_m[1][0][0] = 1
    
    # P2 = x00(y01-y11)
    U_m[2][0][0] = 1
    V_m[2][0][1] = 1; V_m[2][1][1] = -1
    
    # P3 = x11(y10-y00)
    U_m[3][1][1] = 1
    V_m[3][1][0] = 1; V_m[3][0][0] = -1
    
    # P4 = (x00+x01)y11
    U_m[4][0][0] = 1; U_m[4][0][1] = 1
    V_m[4][1][1] = 1
    
    # P5 = (x10-x00)(y00+y01)
    U_m[5][1][0] = 1; U_m[5][0][0] = -1
    V_m[5][0][0] = 1; V_m[5][0][1] = 1
    
    # P6 = (x01-x11)(y10+y11)
    U_m[6][0][1] = 1; U_m[6][1][1] = -1
    V_m[6][1][0] = 1; V_m[6][1][1] = 1
    
    # Reconstruction Gamma 2 (Micro)
    # z00 = P0 + P3 - P4 + P6
    # z01 = P2 + P4
    # z10 = P1 + P3
    # z11 = P0 - P1 + P2 + P5
    gamma_2 = [[[0]*7 for _ in range(2)] for _ in range(2)] # gamma_2[r][c][t]
    
    gamma_2[0][0][0] = 1; gamma_2[0][0][3] = 1; gamma_2[0][0][4] = -1; gamma_2[0][0][6] = 1
    gamma_2[0][1][2] = 1; gamma_2[0][1][4] = 1
    gamma_2[1][0][1] = 1; gamma_2[1][0][3] = 1
    gamma_2[1][1][0] = 1; gamma_2[1][1][1] = -1; gamma_2[1][1][2] = 1; gamma_2[1][1][5] = 1
    
    # Strassen coefficients for 2x2 blocks (Macro level)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # Same structure as micro.
    # alpha_m[s][I][J] coeff of A_IJ in X_s
    # beta_m[s][K][L] coeff of B_KL in Y_s
    
    alpha_m = [[[0]*2 for _ in range(2)] for _ in range(7)]
    beta_m = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # S0
    alpha_m[0][0][0] = 1; alpha_m[0][1][1] = 1
    beta_m[0][0][0] = 1; beta_m[0][1][1] = 1
    
    # S1
    alpha_m[1][1][0] = 1; alpha_m[1][1][1] = 1
    beta_m[1][0][0] = 1
    
    # S2
    alpha_m[2][0][0] = 1
    beta_m[2][0][1] = 1; beta_m[2][1][1] = -1
    
    # S3
    alpha_m[3][1][1] = 1
    beta_m[3][1][0] = 1; beta_m[3][0][0] = -1
    
    # S4
    alpha_m[4][0][0] = 1; alpha_m[4][0][1] = 1
    beta_m[4][1][1] = 1
    
    # S5
    alpha_m[5][1][0] = 1; alpha_m[5][0][0] = -1
    beta_m[5][0][0] = 1; beta_m[5][0][1] = 1
    
    # S6
    alpha_m[6][0][1] = 1; alpha_m[6][1][1] = -1
    beta_m[6][1][0] = 1; beta_m[6][1][1] = 1
    
    # Reconstruction Gamma 1 (Macro)
    # C_block_ij = sum_s gamma_1[i][j][s] P_s
    # Same structure as gamma_2.
    gamma_1 = [[[0]*7 for _ in range(2)] for _ in range(2)] # gamma_1[br][bc][s]
    
    gamma_1[0][0][0] = 1; gamma_1[0][0][3] = 1; gamma_1[0][0][4] = -1; gamma_1[0][0][6] = 1
    gamma_1[0][1][2] = 1; gamma_1[0][1][4] = 1
    gamma_1[1][0][1] = 1; gamma_1[1][0][3] = 1
    gamma_1[1][1][0] = 1; gamma_1[1][1][1] = -1; gamma_1[1][1][2] = 1; gamma_1[1][1][5] = 1
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Update U
            # i = 8*I + 4*a + 2*J + b
            for I in range(2):
                for J in range(2):
                    a_coeff = alpha_m[s][I][J]
                    if a_coeff == 0: continue
                    for a in range(2):
                        for b in range(2):
                            u_coeff = U_m[t][a][b]
                            if u_coeff == 0: continue
                            idx = I * 8 + a * 4 + J * 2 + b
                            U[idx][col] += a_coeff * u_coeff
            
            # Update V
            # j = 8*K + 4*c + 2*L + d
            for K in range(2):
                for L in range(2):
                    b_coeff = beta_m[s][K][L]
                    if b_coeff == 0: continue
                    for c in range(2):
                        for d in range(2):
                            v_coeff = V_m[t][c][d]
                            if v_coeff == 0: continue
                            idx = K * 8 + c * 4 + L * 2 + d
                            V[idx][col] += b_coeff * v_coeff
            
            # Update W
            # k = 8*br + 4*u + 2*bc + v
            for br in range(2):
                for bc in range(2):
                    g1 = gamma_1[br][bc][s]
                    if g1 == 0: continue
                    for u in range(2):
                        for v in range(2):
                            g2 = gamma_2[u][v][t]
                            if g2 == 0: continue
                            idx = br * 8 + u * 4 + bc * 2 + v
                            W[idx][col] += g1 * g2

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Optional: verification
    # import numpy as np
    # A = np.random.randint(0, 2, (4,4))
    # B = np.random.randint(0, 2, (4,4))
    # C = A @ B
    # C_flat = C.flatten()
    # U, V, W = np.array(res["U"]), np.array(res["V"]), np.array(res["W"])
    # A_flat = A.flatten()
    # B_flat = B.flatten()
    # # C_k = sum_i sum_j A_i B_j W_kij ? No, tensor decomposition:
    # # C_k = sum_c U_ic V_jc W_kc ? No.
    # # T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}
    # # Here i indexes A, j indexes B, k indexes C.
    # # C_k = sum_{i,j} T_{i,j,k} A_i B_j = sum_c W_{k,c} (sum_i U_{i,c} A_i) (sum_j V_{j,c} B_j)
    # C_calc = np.zeros(16)
    # for c in range(49):
    #     a_part = sum(A_flat[i] * U[i, c] for i in range(16))
    #     b_part = sum(B_flat[j] * V[j, c] for j in range(16))
    #     for k in range(16):
    #         C_calc[k] += W[k, c] * a_part * b_part
    # if np.allclose(C_calc, C_flat):
    #     print("Verification Passed")
    # else:
    #     print("Verification Failed")
    #     print(C_calc)
    #     print(C_flat)
