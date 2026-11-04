import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_coefs[s][local_i][local_j]
    # V_coefs[s][local_j][local_k]
    # W_coefs[s][local_k][local_i]
    
    # Local indices: i,j,k in {0,1}
    # s in 0..6 corresponding to P1..P7
    
    U_c = [
        [[1,0],[0,0]], # P1: a00
        [[0,0],[1,1]], # P2: a10, a11
        [[1,1],[0,0]], # P3: a00, a01
        [[0,0],[0,1]], # P4: a11
        [[1,0],[0,1]], # P5: a00, a11
        [[0,1],[0,-1]],# P6: a01, -a11
        [[1,0],[-1,0]] # P7: a00, -a10
    ]
    
    V_c = [
        [[1,0],[0,1]], # P1: b00, b11
        [[1,0],[0,0]], # P2: b00
        [[0,0],[0,1]], # P3: b11
        [[-1,0],[1,0]],# P4: -b00, b10
        [[1,1],[0,0]], # P5: b00, b01
        [[0,0],[1,1]], # P6: b10, b11
        [[1,0],[1,0]]  # P7: b00, b10
    ]
    
    # W_coefs[s][local_i][local_k]
    # c00, c01, c10, c11
    # s=0(P1): c01=1, c11=1
    # s=1(P2): c10=1, c11=-1
    # s=2(P3): c00=-1, c01=1
    # s=3(P4): c00=1, c10=1
    # s=4(P5): c00=1, c11=1
    # s=5(P6): c11=1
    # s=6(P7): c00=1
    
    W_c = [
        [[0,1],[0,0]], # P1
        [[0,0],[1,-1]],# P2
        [[-1,1],[0,0]],# P3
        [[1,0],[1,0]], # P4
        [[1,0],[0,1]], # P5
        [[0,0],[0,1]], # P6
        [[1,0],[0,0]]  # P7
    ]
    
    # Blocks: (A_col_start, B_row_start, B_col_start)
    # A is 2x4, B is 4x6
    # A_L: cols 0,1. A_R: cols 2,3
    # B_T: rows 0,1. B_B: rows 2,3
    # B cols split into 0,1; 2,3; 4,5
    
    blocks = [
        (0, 0, 0), # A_L B_T1
        (0, 0, 2), # A_L B_T2
        (0, 0, 4), # A_L B_T3
        (2, 2, 0), # A_R B_B1
        (2, 2, 2), # A_R B_B2
        (2, 2, 4)  # A_R B_B3
    ]
    
    col_offset = 0
    for a_start, b_row_start, b_col_start in blocks:
        for s in range(7):
            # Add Strassen term s to global matrices at col_offset
            for li in range(2):
                for lj in range(2):
                    val = U_c[s][li][lj]
                    if val != 0:
                        global_i = li
                        global_j = a_start + lj
                        U[global_i * m + global_j][col_offset] += val
            
            for lj in range(2):
                for lk in range(2):
                    val = V_c[s][lj][lk]
                    if val != 0:
                        global_j = b_row_start + lj
                        global_k = b_col_start + lk
                        V[global_j * p + global_k][col_offset] += val
            
            for li in range(2):
                for lk in range(2):
                    val = W_c[s][li][lk]
                    if val != 0:
                        global_i = li
                        global_k = b_col_start + lk
                        W[global_k * n + global_i][col_offset] += val
            
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
