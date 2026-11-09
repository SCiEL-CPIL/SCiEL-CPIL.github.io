import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank will be sum of ranks of sub-problems
    # Sub-problems:
    # 1. 2x2x2 (Strassen, rank 7)
    # 2. 2x3x2 (Schoolbook, rank 12)
    # 3. 2x2x3 (Schoolbook, rank 12)
    # 4. 2x3x3 (Schoolbook, rank 18)
    # Total rank 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a sub-decomposition
    def add_sub(n_s, m_s, p_s, U_s, V_s, W_s, i_off, j_off_A, j_off_B, k_off):
        # i_off: offset for i index (always 0 for 2x...)
        # j_off_A: offset for j index in A
        # j_off_B: offset for j index in B
        # k_off: offset for k index
        nonlocal col
        for r in range(len(U_s[0])):
            # U
            for i in range(n_s):
                for j in range(m_s):
                    val = U_s[i * m_s + j][r]
                    if val != 0:
                        U[(i + i_off) * m + (j + j_off_A)][col] += val
            # V
            for j in range(m_s):
                for k in range(p_s):
                    val = V_s[j * p_s + k][r]
                    if val != 0:
                        V[(j + j_off_B) * p + (k + k_off)][col] += val
            # W
            for k in range(p_s):
                for i in range(n_s):
                    val = W_s[k * n_s + i][r]
                    if val != 0:
                        W[(k + k_off) * n + (i + i_off)][col] += val
            col += 1

    # 1. Strassen 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    n_s, m_s, p_s = 2, 2, 2
    R_s = 7
    U_s = [[0] * R_s for _ in range(n_s * m_s)]
    V_s = [[0] * R_s for _ in range(m_s * p_s)]
    W_s = [[0] * R_s for _ in range(p_s * n_s)]
    
    # P1: (A00+A11)(B00+B11)
    # A: 00, 11
    U_s[0][0] = 1; U_s[3][0] = 1
    # B: 00, 11
    V_s[0][0] = 1; V_s[3][0] = 1
    # C: 00 (+1), 11 (+1)
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # P2: (A10+A11)B00
    # A: 10, 11
    U_s[2][1] = 1; U_s[3][1] = 1
    # B: 00
    V_s[0][1] = 1
    # C: 10 (+1), 11 (-1)
    W_s[2][1] = 1; W_s[3][1] = -1
    
    # P3: A00(B01-B11)
    # A: 00
    U_s[0][2] = 1
    # B: 01, 11
    V_s[1][2] = 1; V_s[3][2] = -1
    # C: 01 (+1), 11 (+1)
    W_s[1][2] = 1; W_s[3][2] = 1
    
    # P4: A11(B10-B00)
    # A: 11
    U_s[3][3] = 1
    # B: 10, 00
    V_s[2][3] = 1; V_s[0][3] = -1
    # C: 00 (+1), 10 (+1)
    W_s[0][3] = 1; W_s[2][3] = 1
    
    # P5: (A00+A01)B11
    # A: 00, 01
    U_s[0][4] = 1; U_s[1][4] = 1
    # B: 11
    V_s[3][4] = 1
    # C: 00 (-1), 01 (+1)
    W_s[0][4] = -1; W_s[1][4] = 1
    
    # P6: (A10-A00)(B00+B01)
    # A: 10, 00
    U_s[2][5] = 1; U_s[0][5] = -1
    # B: 00, 01
    V_s[0][5] = 1; V_s[1][5] = 1
    # C: 11 (+1)
    W_s[3][5] = 1
    
    # P7: (A01-A11)(B10+B11)
    # A: 01, 11
    U_s[1][6] = 1; U_s[3][6] = -1
    # B: 10, 11
    V_s[2][6] = 1; V_s[3][6] = 1
    # C: 00 (+1)
    W_s[0][6] = 1
    
    # Add Sub 1: A cols 0-1, B rows 0-1, C cols 0-1
    add_sub(2, 2, 2, U_s, V_s, W_s, 0, 0, 0, 0)
    
    # 2. Schoolbook 2x3x2
    n_s, m_s, p_s = 2, 3, 2
    R_s = n_s * m_s * p_s
    U_s = [[0] * R_s for _ in range(n_s * m_s)]
    V_s = [[0] * R_s for _ in range(m_s * p_s)]
    W_s = [[0] * R_s for _ in range(p_s * n_s)]
    c = 0
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                U_s[i * m_s + j][c] = 1
                V_s[j * p_s + k][c] = 1
                W_s[k * n_s + i][c] = 1
                c += 1
    # Add Sub 2: A cols 2-4, B rows 2-4, C cols 0-1
    add_sub(2, 3, 2, U_s, V_s, W_s, 0, 2, 2, 0)
    
    # 3. Schoolbook 2x2x3
    n_s, m_s, p_s = 2, 2, 3
    R_s = n_s * m_s * p_s
    U_s = [[0] * R_s for _ in range(n_s * m_s)]
    V_s = [[0] * R_s for _ in range(m_s * p_s)]
    W_s = [[0] * R_s for _ in range(p_s * n_s)]
    c = 0
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                U_s[i * m_s + j][c] = 1
                V_s[j * p_s + k][c] = 1
                W_s[k * n_s + i][c] = 1
                c += 1
    # Add Sub 3: A cols 0-1, B rows 0-1, C cols 2-4
    add_sub(2, 2, 3, U_s, V_s, W_s, 0, 0, 0, 2)
    
    # 4. Schoolbook 2x3x3
    n_s, m_s, p_s = 2, 3, 3
    R_s = n_s * m_s * p_s
    U_s = [[0] * R_s for _ in range(n_s * m_s)]
    V_s = [[0] * R_s for _ in range(m_s * p_s)]
    W_s = [[0] * R_s for _ in range(p_s * n_s)]
    c = 0
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                U_s[i * m_s + j][c] = 1
                V_s[j * p_s + k][c] = 1
                W_s[k * n_s + i][c] = 1
                c += 1
    # Add Sub 4: A cols 2-4, B rows 2-4, C cols 2-4
    add_sub(2, 3, 3, U_s, V_s, W_s, 0, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
