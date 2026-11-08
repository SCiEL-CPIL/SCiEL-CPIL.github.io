import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 8
    
    # Rank 7 Strassen decomposition for 2x2x2
    # U2[i*2+j], V2[j*2+k], W2[k*2+i]
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1; S_W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1; S_W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = -1; S_W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1; S_W[3, 6] = 1
    
    # Rank 49 decomposition for 4x4x4 using recursive Strassen
    # View 4x4 as 2x2 blocks of 2x2 matrices
    # Outer Strassen: 7 terms for the block-level multiplication
    # Inner Strassen: 7 terms for each 2x2 block product
    
    # Outer Strassen coefficients for 4x4 (blocks indexed 0,1)
    # A_outer[t][I][J] = coefficient of block A[I,J] in term t
    A_outer = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[-1, 0], [1, 0]],  # M6: A10 - A00
        [[0, 1], [0, -1]],  # M7: A01 - A11
    ]
    
    B_outer = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01 - B11
        [[0, -1], [1, 0]],  # M4: B10 - B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 1], [0, 1]],   # M7: B10 + B11
    ]
    
    C_outer = [
        [[1, 0], [0, 1]],   # M1: C00 + C11
        [[0, 0], [1, 1]],   # M2: C10 + C11
        [[0, 1], [0, 1]],   # M3: C01 + C11
        [[1, 0], [0, 1]],   # M4: C00 + C10
        [[-1, 1], [0, 1]],  # M5: -C00 + C01 + C11
        [[1, 0], [0, 0]],   # M6: C00
        [[0, 0], [0, 1]],   # M7: C11
    ]
    
    R = 98
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    for half in range(2):
        k_offset = half * 4
        base_col = half * 49
        
        for t1 in range(7):  # Outer Strassen term
            for t2 in range(7):  # Inner Strassen term
                col = base_col + t1 * 7 + t2
                
                # U contribution
                for I in range(2):
                    for J in range(2):
                        for i_l in range(2):
                            for j_l in range(2):
                                i = I * 2 + i_l
                                j = J * 2 + j_l
                                coeff = A_outer[t1][I][J] * S_U[i_l * 2 + j_l, t2]
                                if coeff != 0:
                                    U[i * 4 + j, col] += coeff
                
                # V contribution
                for K in range(2):
                    for J in range(2):
                        for j_l in range(2):
                            for k_l in range(2):
                                j = K * 2 + j_l
                                k = k_offset + J * 2 + k_l
                                coeff = B_outer[t1][K][J] * S_V[j_l * 2 + k_l, t2]
                                if coeff != 0:
                                    V[j * 8 + k, col] += coeff
                
                # W contribution
                for I in range(2):
                    for J in range(2):
                        for i_l in range(2):
                            for k_l in range(2):
                                i = I * 2 + i_l
                                k = k_offset + J * 2 + k_l
                                coeff = C_outer[t1][I][J] * S_W[i_l * 2 + k_l, t2]
                                if coeff != 0:
                                    W[k * 4 + i, col] += coeff
    
    if time.time() > deadline:
        # Fallback to schoolbook if we're running out of time
        rank = 128
        U_fb = np.zeros((n * m, rank), dtype=float)
        V_fb = np.zeros((m * p, rank), dtype=float)
        W_fb = np.zeros((p * n, rank), dtype=float)
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U_fb[i * 4 + j, column] = 1
                    V_fb[j * 8 + k, column] = 1
                    W_fb[k * 4 + i, column] = 1
                    column += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
