"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # rank = 88
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, coeff=1):
        nonlocal col
        if col >= rank:
            return
        for i in range(n * m):
            U[i][col] += u_vec[i] * coeff
        for i in range(m * p):
            V[i][col] += v_vec[i] * coeff
        for i in range(p * n):
            W[i][col] += w_vec[i] * coeff
        col += 1

    # Iterate over 2 parts of A (i_part) and 4 parts of B (j_part)
    for i_part in range(2):
        # A rows are 0, 1, 2
        # A cols for this part: 2*i_part, 2*i_part+1
        # B rows for this part: 2*i_part, 2*i_part+1
        for j_part in range(4):
            # B cols for this part: 2*j_part, 2*j_part+1
            # C rows (from B cols): 2*j_part, 2*j_part+1
            # C cols (from A rows): 0, 1, 2
            
            # 1. Decompose mu_1,2,2 for A row 0
            # A index for row 0: 0*m + 2*i_part, 0*m + 2*i_part+1
            a_idx_1_1 = 0 * m + 2 * i_part
            a_idx_1_2 = 0 * m + 2 * i_part + 1
            
            # B indices
            # y_11: row 2*i_part, col 2*j_part
            b_idx_11 = (2 * i_part) * p + 2 * j_part
            # y_12: row 2*i_part, col 2*j_part+1
            b_idx_12 = (2 * i_part) * p + 2 * j_part + 1
            # y_21: row 2*i_part+1, col 2*j_part
            b_idx_21 = (2 * i_part + 1) * p + 2 * j_part
            # y_22: row 2*i_part+1, col 2*j_part+1
            b_idx_22 = (2 * i_part + 1) * p + 2 * j_part + 1
            
            # C indices
            # z_1: row 2*j_part, col 0
            c_idx_1 = (2 * j_part) * n + 0
            # z_2: row 2*j_part+1, col 0
            c_idx_2 = (2 * j_part + 1) * n + 0
            
            u_1 = [0] * (n * m); u_1[a_idx_1_1] = 1
            u_2 = [0] * (n * m); u_2[a_idx_1_2] = 1
            
            v_11 = [0] * (m * p); v_11[b_idx_11] = 1
            v_12 = [0] * (m * p); v_12[b_idx_12] = 1
            v_21 = [0] * (m * p); v_21[b_idx_21] = 1
            v_22 = [0] * (m * p); v_22[b_idx_22] = 1
            
            w_1 = [0] * (p * n); w_1[c_idx_1] = 1
            w_2 = [0] * (p * n); w_2[c_idx_2] = 1
            
            # Term 1: x1 y11 -> z1
            add_term(u_1, v_11, w_1)
            # Term 2: x2 y21 -> z1
            add_term(u_2, v_21, w_1)
            # Term 3: x1 y12 -> z2
            add_term(u_1, v_12, w_2)
            # Term 4: x2 y22 -> z2
            add_term(u_2, v_22, w_2)
            
            # 2. Decompose mu_2,2,2 for A rows 1, 2
            # A indices
            # Row 1: 1*m + 2*i_part, 1*m + 2*i_part+1
            # Row 2: 2*m + 2*i_part, 2*m + 2*i_part+1
            a_11 = 1 * m + 2 * i_part
            a_12 = 1 * m + 2 * i_part + 1
            a_21 = 2 * m + 2 * i_part
            a_22 = 2 * m + 2 * i_part + 1
            
            # B indices are same as above (b_idx_11, etc)
            
            # C indices
            # z_11: row 2*j_part, col 1
            c_11 = (2 * j_part) * n + 1
            # z_12: row 2*j_part+1, col 1
            c_12 = (2 * j_part + 1) * n + 1
            # z_21: row 2*j_part, col 2
            c_21 = (2 * j_part) * n + 2
            # z_22: row 2*j_part+1, col 2
            c_22 = (2 * j_part + 1) * n + 2
            
            # Strassen terms
            # M1 = (x11+x22)(y11+y22)
            u_M1 = [0] * (n * m); u_M1[a_11] = 1; u_M1[a_22] = 1
            v_M1 = [0] * (m * p); v_M1[b_idx_11] = 1; v_M1[b_idx_22] = 1
            w_M1 = [0] * (p * n); w_M1[c_11] = 1; w_M1[c_22] = 1
            add_term(u_M1, v_M1, w_M1)
            
            # M2 = (x21+x22)y11
            u_M2 = [0] * (n * m); u_M2[a_21] = 1; u_M2[a_22] = 1
            v_M2 = [0] * (m * p); v_M2[b_idx_11] = 1
            w_M2 = [0] * (p * n); w_M2[c_21] = 1; w_M2[c_22] = -1
            add_term(u_M2, v_M2, w_M2)
            
            # M3 = x11(y12-y22)
            u_M3 = [0] * (n * m); u_M3[a_11] = 1
            v_M3 = [0] * (m * p); v_M3[b_idx_12] = 1; v_M3[b_idx_22] = -1
            w_M3 = [0] * (p * n); w_M3[c_12] = 1; w_M3[c_22] = 1
            add_term(u_M3, v_M3, w_M3)
            
            # M4 = x22(y21-y11)
            u_M4 = [0] * (n * m); u_M4[a_22] = 1
            v_M4 = [0] * (m * p); v_M4[b_idx_21] = 1; v_M4[b_idx_11] = -1
            w_M4 = [0] * (p * n); w_M4[c_11] = 1; w_M4[c_21] = 1
            add_term(u_M4, v_M4, w_M4)
            
            # M5 = (x11+x12)y22
            u_M5 = [0] * (n * m); u_M5[a_11] = 1; u_M5[a_12] = 1
            v_M5 = [0] * (m * p); v_M5[b_idx_22] = 1
            w_M5 = [0] * (p * n); w_M5[c_11] = -1; w_M5[c_12] = 1
            add_term(u_M5, v_M5, w_M5)
            
            # M6 = (x21-x11)(y11+y12)
            u_M6 = [0] * (n * m); u_M6[a_21] = 1; u_M6[a_11] = -1
            v_M6 = [0] * (m * p); v_M6[b_idx_11] = 1; v_M6[b_idx_12] = 1
            w_M6 = [0] * (p * n); w_M6[c_22] = 1
            add_term(u_M6, v_M6, w_M6)
            
            # M7 = (x12-x22)(y21+y22)
            u_M7 = [0] * (n * m); u_M7[a_12] = 1; u_M7[a_22] = -1
            v_M7 = [0] * (m * p); v_M7[b_idx_21] = 1; v_M7[b_idx_22] = 1
            w_M7 = [0] * (p * n); w_M7[c_11] = 1
            add_term(u_M7, v_M7, w_M7)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
