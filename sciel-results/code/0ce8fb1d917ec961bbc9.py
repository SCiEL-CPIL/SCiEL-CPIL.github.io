import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin

    # 1. Strassen 2x2 Decomposition (Rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: A00+A11, B00+B11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2: A10+A11, B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # M3: A00, B01-B11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # M4: A11, B10-B00
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # M5: A00+A01, B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # M6: A10-A00, B00+B01
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7: A01-A11, B10+B11
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W2 Output Reconstruction (Trace convention W[k*n+i])
    # C00 = M1 + M4 - M5 + M7
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C01 = M3 + M5
    W2[2, 2] = 1; W2[2, 4] = 1
    # C10 = M2 + M4
    W2[1, 1] = 1; W2[1, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1

    # 2. Laderman 3x3 Decomposition (Rank 23)
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # Laderman's 23-term algorithm for 3x3 matrix multiplication
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Term 0: M1
    U3[0, 0] = 1; U3[4, 0] = 1  # A00 + A11
    V3[0, 0] = 1; V3[4, 0] = 1  # B00 + B11
    W3[0, 0] = 1; W3[4, 0] = 1; W3[1, 0] = -1; W3[3, 0] = -1  # C00 + C11 - C01 - C10
    
    # Term 1: M2
    U3[0, 1] = 1; U3[4, 1] = -1  # A00 - A11
    V3[3, 1] = 1; V3[4, 1] = -1  # B10 - B11
    W3[3, 1] = 1; W3[4, 1] = 1; W3[0, 1] = -1; W3[1, 1] = -1  # C10 + C11 - C00 - C01
    
    # Term 2: M3
    U3[0, 2] = 1; U3[1, 2] = 1  # A00 + A01
    V3[3, 2] = 1; V3[4, 2] = 1  # B10 + B11
    W3[0, 2] = 1; W3[3, 2] = 1; W3[4, 2] = -1; W3[1, 2] = -1  # C00 + C10 - C11 - C01
    
    # Term 3: M4
    U3[1, 3] = 1; U3[4, 3] = -1  # A01 - A11
    V3[0, 3] = 1; V3[4, 3] = 1  # B00 + B11
    W3[1, 3] = 1; W3[4, 3] = 1; W3[0, 3] = -1; W3[3, 3] = -1  # C01 + C11 - C00 - C10
    
    # Term 4: M5
    U3[3, 4] = 1; U3[4, 4] = 1  # A10 + A11
    V3[1, 4] = 1; V3[4, 4] = 1  # B01 + B11
    W3[3, 4] = 1; W3[4, 4] = 1; W3[0, 4] = -1; W3[1, 4] = -1  # C10 + C11 - C00 - C01
    
    # Term 5: M6
    U3[3, 5] = 1; U3[0, 5] = -1  # A10 - A00
    V3[0, 5] = 1; V3[1, 5] = 1  # B00 + B01
    W3[0, 5] = 1; W3[3, 5] = 1; W3[4, 5] = -1; W3[1, 5] = -1  # C00 + C10 - C11 - C01
    
    # Term 6: M7
    U3[1, 6] = 1; U3[4, 6] = -1  # A01 - A11
    V3[3, 6] = 1; V3[4, 6] = 1  # B10 + B11
    W3[1, 6] = 1; W3[3, 6] = 1; W3[4, 6] = -1; W3[0, 6] = -1  # C01 + C10 - C11 - C00
    
    # Term 7: M8
    U3[6, 7] = 1; U3[8, 7] = 1  # A20 + A22
    V3[0, 7] = 1; V3[8, 7] = 1  # B00 + B22
    W3[6, 7] = 1; W3[8, 7] = 1; W3[7, 7] = -1; W3[5, 7] = -1  # C20 + C22 - C21 - C02
    
    # Term 8: M9
    U3[6, 8] = 1; U3[8, 8] = -1  # A20 - A22
    V3[6, 8] = 1; V3[8, 8] = -1  # B20 - B22
    W3[5, 8] = 1; W3[8, 8] = 1; W3[6, 8] = -1; W3[7, 8] = -1  # C02 + C22 - C20 - C21
    
    # Term 9: M10
    U3[6, 9] = 1; U3[7, 9] = 1  # A20 + A21
    V3[6, 9] = 1; V3[8, 9] = 1  # B20 + B22
    W3[6, 9] = 1; W3[5, 9] = 1; W3[8, 9] = -1; W3[7, 9] = -1  # C20 + C02 - C22 - C21
    
    # Term 10: M11
    U3[7, 10] = 1; U3[8, 10] = -1  # A21 - A22
    V3[0, 10] = 1; V3[8, 10] = 1  # B00 + B22
    W3[7, 10] = 1; W3[8, 10] = 1; W3[6, 10] = -1; W3[5, 10] = -1  # C21 + C22 - C20 - C02
    
    # Term 11: M12
    U3[2, 11] = 1; U3[5, 11] = 1  # A02 + A12
    V3[0, 11] = 1; V3[4, 11] = 1  # B00 + B11
    W3[2, 11] = 1; W3[5, 11] = 1; W3[1, 11] = -1; W3[4, 11] = -1  # C02 + C12 - C01 - C11
    
    # Term 12: M13
    U3[2, 12] = 1; U3[5, 12] = -1  # A02 - A12
    V3[3, 12] = 1; V3[4, 12] = -1  # B10 - B11
    W3[4, 12] = 1; W3[5, 12] = 1; W3[2, 12] = -1; W3[1, 12] = -1  # C11 + C12 - C02 - C01
    
    # Term 13: M14
    U3[2, 13] = 1; U3[1, 13] = 1  # A02 + A01
    V3[3, 13] = 1; V3[4, 13] = 1  # B10 + B11
    W3[2, 13] = 1; W3[4, 13] = 1; W3[5, 13] = -1; W3[1, 13] = -1  # C02 + C11 - C12 - C01
    
    # Term 14: M15
    U3[1, 14] = 1; U3[5, 14] = -1  # A01 - A12
    V3[0, 14] = 1; V3[4, 14] = 1  # B00 + B11
    W3[1, 14] = 1; W3[5, 14] = 1; W3[2, 14] = -1; W3[4, 14] = -1  # C01 + C12 - C02 - C11
    
    # Term 15: M16
    U3[3, 15] = 1; U3[5, 15] = 1  # A10 + A12
    V3[1, 15] = 1; V3[4, 15] = 1  # B01 + B11
    W3[3, 15] = 1; W3[5, 15] = 1; W3[2, 15] = -1; W3[1, 15] = -1  # C10 + C12 - C02 - C01
    
    # Term 16: M17
    U3[3, 16] = 1; U3[2, 16] = -1  # A10 - A02
    V3[0, 16] = 1; V3[1, 16] = 1  # B00 + B01
    W3[2, 16] = 1; W3[3, 16] = 1; W3[5, 16] = -1; W3[1, 16] = -1  # C02 + C10 - C12 - C01
    
    # Term 17: M18
    U3[1, 17] = 1; U3[5, 17] = -1  # A01 - A12
    V3[3, 17] = 1; V3[4, 17] = 1  # B10 + B11
    W3[1, 17] = 1; W3[3, 17] = 1; W3[5, 17] = -1; W3[2, 17] = -1  # C01 + C10 - C12 - C02
    
    # Term 18: M19
    U3[8, 18] = 1  # A22
    V3[8, 18] = 1  # B22
    W3[8, 18] = 1  # C22
    
    # Term 19: M20
    U3[8, 19] = 1  # A22
    V3[0, 19] = 1  # B00
    W3[6, 19] = 1  # C20
    
    # Term 20: M21
    U3[0, 20] = 1  # A00
    V3[8, 20] = 1  # B22
    W3[2, 20] = 1  # C02
    
    # Term 21: M22
    U3[6, 21] = 1  # A20
    V3[2, 21] = 1  # B02
    W3[4, 21] = 1  # C12
    
    # Term 22: M23
    U3[2, 22] = 1  # A02
    V3[6, 22] = 1  # B20
    W3[0, 22] = 1  # C00

    # Verify Laderman 3x3
    def verify_3x3(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    use_laderman = verify_3x3(U3, V3, W3)
    
    if not use_laderman:
        # Fallback to Schoolbook 3x3 (Rank 27)
        U3 = np.zeros((9, 27))
        V3 = np.zeros((9, 27))
        W3 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U3[i*3+j, col] = 1
                    V3[j*3+k, col] = 1
                    W3[k*3+i, col] = 1
                    col += 1

    # 3. Kronecker Product Construction
    rank2 = 7
    rank3 = U3.shape[1]
    rank_total = rank2 * rank3
    
    U6 = np.zeros((36, rank_total))
    V6 = np.zeros((36, rank_total))
    W6 = np.zeros((36, rank_total))
    
    for r3 in range(rank3):
        for r2 in range(rank2):
            r = r3 * rank2 + r2
            
            # U Part
            for i3 in range(3):
                for j3 in range(3):
                    u3_val = U3[i3*3+j3, r3]
                    if u3_val != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                u2_val = U2[i2*2+j2, r2]
                                if u2_val != 0:
                                    i = i3*2 + i2
                                    j = j3*2 + j2
                                    idx = i*6 + j
                                    U6[idx, r] = u3_val * u2_val
                                    
            # V Part
            for j3 in range(3):
                for k3 in range(3):
                    v3_val = V3[j3*3+k3, r3]
                    if v3_val != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                v2_val = V2[j2*2+k2, r2]
                                if v2_val != 0:
                                    j = j3*2 + j2
                                    k = k3*2 + k2
                                    idx = j*6 + k
                                    V6[idx, r] = v3_val * v2_val
                                    
            # W Part
            for k3 in range(3):
                for i3 in range(3):
                    w3_val = W3[k3*3+i3, r3]
                    if w3_val != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                w2_val = W2[k2*2+i2, r2]
                                if w2_val != 0:
                                    k = k3*2 + k2
                                    i = i3*2 + i2
                                    idx = k*6 + i
                                    W6[idx, r] = w3_val * w2_val
                                    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
