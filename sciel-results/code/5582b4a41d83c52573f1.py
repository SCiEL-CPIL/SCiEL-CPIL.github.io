import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices: U: i*2+j, V: j*2+k, W: k*2+i
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # Term 0: m1 = (A00+A11)(B00+B11) -> C00 + C11
    U_S[0][0] = 1; U_S[3][0] = 1
    V_S[0][0] = 1; V_S[3][0] = 1
    W_S[0][0] = 1; W_S[3][0] = 1
    
    # Term 1: m2 = (A10+A11)B00 -> C10 + C11 (coeff 1, -1)
    U_S[2][1] = 1; U_S[3][1] = 1
    V_S[0][1] = 1
    W_S[1][1] = 1; W_S[3][1] = -1
    
    # Term 2: m3 = A00(B10-B11) -> C01 + C11
    U_S[0][2] = 1
    V_S[2][2] = 1; V_S[3][2] = -1
    W_S[2][2] = 1; W_S[3][2] = 1
    
    # Term 3: m4 = A11(B01-B10) -> C00 + C10
    U_S[3][3] = 1
    V_S[1][3] = 1; V_S[2][3] = -1
    W_S[0][3] = 1; W_S[1][3] = 1
    
    # Term 4: m5 = (A00+A01)B11 -> -C00 + C01
    U_S[0][4] = 1; U_S[1][4] = 1
    V_S[3][4] = 1
    W_S[0][4] = -1; W_S[2][4] = 1
    
    # Term 5: m6 = (A10-A00)(B00+B01) -> C11
    U_S[2][5] = 1; U_S[0][5] = -1
    V_S[0][5] = 1; V_S[1][5] = 1
    W_S[3][5] = 1
    
    # Term 6: m7 = (A01-A11)(B10+B11) -> -C00
    U_S[1][6] = 1; U_S[3][6] = -1
    V_S[2][6] = 1; V_S[3][6] = 1
    W_S[0][6] = -1

    # Schoolbook 3x3
    # U_SB: 9x27, V_SB: 9x27, W_SB: 9x27
    # Indices: U: i*3+j, V: j*3+k, W: k*3+i
    U_SB = [[0]*27 for _ in range(9)]
    V_SB = [[0]*27 for _ in range(9)]
    W_SB = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_SB[i*3+j][col] = 1
                V_SB[j*3+k][col] = 1
                W_SB[k*3+i][col] = 1
                col += 1

    # Construct 6x6 matrices
    # Rank = 7 * 27 = 189
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Precompute sparse representations
    # For each term r, list of (index, value)
    U_S_sp = []
    for r in range(7):
        sp = []
        for idx in range(4):
            if U_S[idx][r] != 0:
                sp.append((idx, U_S[idx][r]))
        U_S_sp.append(sp)
        
    V_S_sp = []
    for r in range(7):
        sp = []
        for idx in range(4):
            if V_S[idx][r] != 0:
                sp.append((idx, V_S[idx][r]))
        V_S_sp.append(sp)
        
    W_S_sp = []
    for r in range(7):
        sp = []
        for idx in range(4):
            if W_S[idx][r] != 0:
                sp.append((idx, W_S[idx][r]))
        W_S_sp.append(sp)

    U_SB_sp = []
    for r in range(27):
        sp = []
        for idx in range(9):
            if U_SB[idx][r] != 0:
                sp.append((idx, U_SB[idx][r]))
        U_SB_sp.append(sp)
        
    V_SB_sp = []
    for r in range(27):
        sp = []
        for idx in range(9):
            if V_SB[idx][r] != 0:
                sp.append((idx, V_SB[idx][r]))
        V_SB_sp.append(sp)
        
    W_SB_sp = []
    for r in range(27):
        sp = []
        for idx in range(9):
            if W_SB[idx][r] != 0:
                sp.append((idx, W_SB[idx][r]))
        W_SB_sp.append(sp)

    # Mapping functions
    def map_U(idx_S, idx_SB):
        # idx_S = i_S*2 + j_S
        # idx_SB = i_SB*3 + j_SB
        i_S = idx_S // 2
        j_S = idx_S % 2
        i_SB = idx_SB // 3
        j_SB = idx_SB % 3
        # I = i*6 + j = (i_S*3 + i_SB)*6 + (j_S*3 + j_SB)
        return (i_S * 3 + i_SB) * 6 + (j_S * 3 + j_SB)

    def map_V(idx_S, idx_SB):
        # idx_S = j_S*2 + k_S
        # idx_SB = j_SB*3 + k_SB
        j_S = idx_S // 2
        k_S = idx_S % 2
        j_SB = idx_SB // 3
        k_SB = idx_SB % 3
        # J = j*6 + k = (j_S*3 + j_SB)*6 + (k_S*3 + k_SB)
        return (j_S * 3 + j_SB) * 6 + (k_S * 3 + k_SB)

    def map_W(idx_S, idx_SB):
        # idx_S = k_S*2 + i_S
        # idx_SB = k_SB*3 + i_SB
        k_S = idx_S // 2
        i_S = idx_S % 2
        k_SB = idx_SB // 3
        i_SB = idx_SB % 3
        # L = k*6 + i = (k_S*3 + k_SB)*6 + (i_S*3 + i_SB)
        return (k_S * 3 + k_SB) * 6 + (i_S * 3 + i_SB)

    # Fill matrices
    for r_S in range(7):
        for r_SB in range(27):
            r = r_S * 27 + r_SB
            
            for idx_S, val_S in U_S_sp[r_S]:
                for idx_SB, val_SB in U_SB_sp[r_SB]:
                    I = map_U(idx_S, idx_SB)
                    U[I][r] += val_S * val_SB
                    
            for idx_S, val_S in V_S_sp[r_S]:
                for idx_SB, val_SB in V_SB_sp[r_SB]:
                    J = map_V(idx_S, idx_SB)
                    V[J][r] += val_S * val_SB
                    
            for idx_S, val_S in W_S_sp[r_S]:
                for idx_SB, val_SB in W_SB_sp[r_SB]:
                    L = map_W(idx_S, idx_SB)
                    W[L][r] += val_S * val_SB

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
