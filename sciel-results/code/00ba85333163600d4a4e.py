import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # T0 = (x11 + x22)(y11 + y22)
    # T1 = (x21 + x22)y11
    # T2 = x11(y12 - y22)
    # T3 = x22(y21 - y11)
    # T4 = (x11 + x12)y22
    # T5 = (x21 - x11)(y11 + y12)
    # T6 = (x12 - x22)(y21 + y22)
    
    # U_sub[t][i,j] is coeff of x_ij in U_t
    # V_sub[t][i,j] is coeff of y_ij in V_t
    U_sub = [
        [[1, 0], [0, 1]],   # T0: x11 + x22
        [[0, 0], [1, 1]],   # T1: x21 + x22
        [[1, 0], [0, 0]],   # T2: x11
        [[0, 0], [0, 1]],   # T3: x22
        [[1, 1], [0, 0]],   # T4: x11 + x12
        [[-1, 0], [1, 0]],  # T5: x21 - x11
        [[0, 1], [0, -1]]   # T6: x12 - x22
    ]
    
    V_sub = [
        [[1, 0], [0, 1]],   # T0: y11 + y22
        [[1, 0], [0, 0]],   # T1: y11
        [[0, 1], [0, -1]],  # T2: y12 - y22
        [[0, -1], [1, 0]],  # T3: y21 - y11
        [[0, 0], [0, 1]],   # T4: y22
        [[1, 1], [0, 0]],   # T5: y11 + y12
        [[0, 0], [1, 1]]    # T6: y21 + y22
    ]
    
    # W_sub[t][i,j] is coeff of T_t in Z_ij
    W_sub = [
        [[1, 0], [0, 1]],   # Z11: T0 + T3 - T4 + T6 -> T0:1, T3:1, T4:-1, T6:1
        [[0, 1], [0, 0]],   # Z12: T2 + T4 -> T2:1, T4:1
        [[0, 0], [1, 0]],   # Z21: T1 + T3 -> T1:1, T3:1
        [[1, 0], [0, 1]]    # Z22: T0 - T1 + T2 + T5 -> T0:1, T1:-1, T2:1, T5:1
    ]
    # Correct W_sub structure: W_sub[i,j][t] is coeff of T_t in Z_ij
    W_sub_correct = [
        [[1, 0, 0, 1, -1, 0, 1]],  # Z11
        [[0, 0, 1, 0, 1, 0, 0]],   # Z12
        [[0, 1, 0, 1, 0, 0, 0]],   # Z21
        [[1, -1, 1, 0, 0, 1, 0]]   # Z22
    ]
    
    # Block Strassen coefficients for 4x4 matrix (viewed as 2x2 blocks)
    # M0 = (A11 + A22)(B11 + B22)
    # M1 = (A21 + A22)B11
    # M2 = A11(B12 - B22)
    # M3 = A22(B21 - B11)
    # M4 = (A11 + A12)B22
    # M5 = (A21 - A11)(B11 + B12)
    # M6 = (A12 - A22)(B21 + B22)
    
    # A_block[s][i,j] is coeff of A_block_ij in X_s
    # B_block[s][i,j] is coeff of B_block_ij in Y_s
    A_block = [
        [[1, 0], [0, 1]],   # M0: A11 + A22
        [[0, 0], [1, 1]],   # M1: A21 + A22
        [[1, 0], [0, 0]],   # M2: A11
        [[0, 0], [0, 1]],   # M3: A22
        [[1, 1], [0, 0]],   # M4: A11 + A12
        [[-1, 0], [1, 0]],  # M5: A21 - A11
        [[0, 1], [0, -1]]   # M6: A12 - A22
    ]
    
    B_block = [
        [[1, 0], [0, 1]],   # M0: B11 + B22
        [[1, 0], [0, 0]],   # M1: B11
        [[0, 1], [0, -1]],  # M2: B12 - B22
        [[0, -1], [1, 0]],  # M3: B21 - B11
        [[0, 0], [0, 1]],   # M4: B22
        [[1, 1], [0, 0]],   # M5: B11 + B12
        [[0, 0], [1, 1]]    # M6: B21 + B22
    ]
    
    # C_block[s][i,j] is coeff of M_s in C_block_ij
    # C11 = M0 + M3 - M4 + M6
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M1 + M2 + M5
    C_block = [
        [[1, 0, 0, 1, -1, 0, 1]],  # C11
        [[0, 0, 1, 0, 1, 0, 0]],   # C12
        [[0, 1, 0, 1, 0, 0, 0]],   # C21
        [[1, -1, 1, 0, 0, 1, 0]]   # C22
    ]
    
    # Map block indices to 4x4 indices
    # Block (bi, bj) covers i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Determine U coefficients
            # U is a linear combination of A's elements
            # U_t(X_s) = sum_{x_i, x_j} U_sub[t][x_i][x_j] * X_s[x_i][x_j]
            # X_s[x_i][x_j] = sum_{a_i, a_j} A_block[s][x_i][x_j] * A_block[a_i][a_j]
            # So U_t(X_s) = sum_{a_i, a_j} (sum_{x_i, x_j} U_sub[t][x_i][x_j] * A_block[s][x_i][x_j]) * A_block[a_i][a_j]
            
            U_coeff = [[0.0] * 4 for _ in range(4)]
            for xi in range(2):
                for xj in range(2):
                    c_xt = U_sub[t][xi][xj]
                    if c_xt == 0: continue
                    for ai in range(2):
                        for aj in range(2):
                            c_ab = A_block[s][ai][aj]
                            if c_ab == 0: continue
                            coeff = c_xt * c_ab
                            if coeff != 0:
                                # Map block (ai, aj) to 4x4 indices
                                for i in range(2 * ai, 2 * ai + 2):
                                    for j in range(2 * aj, 2 * aj + 2):
                                        U[i * m + j][col] += coeff
                                        
            # Determine V coefficients
            V_coeff = [[0.0] * 4 for _ in range(4)]
            for yi in range(2):
                for yj in range(2):
                    c_yt = V_sub[t][yi][yj]
                    if c_yt == 0: continue
                    for bi in range(2):
                        for bj in range(2):
                            c_bb = B_block[s][bi][bj]
                            if c_bb == 0: continue
                            coeff = c_yt * c_bb
                            if coeff != 0:
                                for i in range(2 * bi, 2 * bi + 2):
                                    for j in range(2 * bj, 2 * bj + 2):
                                        V[i * m + j][col] += coeff
                                        
            # Determine W coefficients
            # W is the coefficient for the output C
            # The term is U_t(X_s) V_t(Y_s)
            # This contributes to Z_s = X_s Y_s
            # Z_s[xi][xj] has coefficient W_sub_correct[xi][xj][t] from T_t
            # Z_s[xi][xj] contributes to C_block[ci][cj] with coefficient C_block[ci][cj][s]
            # So the total coefficient for C_block[ci][cj] is sum_{xi, xj} W_sub_correct[xi][xj][t] * C_block[ci][cj][s]
            
            for ci in range(2):
                for cj in range(2):
                    c_cs = C_block[ci][cj][s]
                    if c_cs == 0: continue
                    total_w = 0.0
                    for xi in range(2):
                        for xj in range(2):
                            c_wt = W_sub_correct[xi][xj][t]
                            total_w += c_wt * c_cs
                    if total_w != 0:
                        for i in range(2 * ci, 2 * ci + 2):
                            for k in range(2 * cj, 2 * cj + 2):
                                W[k * n + i][col] += total_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
