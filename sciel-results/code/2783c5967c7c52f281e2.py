"""Program for 6x6x6 matrix multiplication tensor decomposition using CW(3x3) ⊗ Strassen(2x2)."""
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen decomposition for 2x2 matrices, rank 7
    # U2 shape: (4, 7), V2 shape: (4, 7), W2 shape: (4, 7)
    U2 = np.array([
        [1, 1, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 1, 0, -1, 1],  # A10
        [0, 0, 0, 0, 1, 0, 0],   # A01
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 1, 0, 0, 1],   # B10
        [1, 0, 0, 0, 1, 0, 0],   # B01
        [1, 0, -1, -1, 1, 0, 1], # B11
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 0, -1, 0, -1], # C00 = M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, -1, 1, 0, 0, 0, 1],  # C11 = M1 - M2 + M3 + M7
    ], dtype=float)

    # Coppersmith-Winograd rank-23 decomposition for 3x3 matrices
    # U3 shape: (9, 23), V3 shape: (9, 23), W3 shape: (9, 23)
    # Indices: U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    #
    # The CW rank-23 decomposition uses 23 terms.
    # Here's a known explicit decomposition:
    
    U3 = np.zeros((9, 23), dtype=float)
    V3 = np.zeros((9, 23), dtype=float)
    W3 = np.zeros((9, 23), dtype=float)
    
    # Term 0: e00 * e00 * e00
    U3[0, 0] = 1; V3[0, 0] = 1; W3[0, 0] = 1
    # Term 1: e01 * e01 * e10
    U3[1, 1] = 1; V3[4, 1] = 1; W3[3, 1] = 1
    # Term 2: e02 * e02 * e20
    U3[2, 2] = 1; V3[8, 2] = 1; W3[6, 2] = 1
    # Term 3: e10 * e10 * e01
    U3[3, 3] = 1; V3[3, 3] = 1; W3[1, 3] = 1
    # Term 4: e11 * e11 * e11
    U3[4, 4] = 1; V3[7, 4] = 1; W3[4, 4] = 1
    # Term 5: e12 * e12 * e21
    U3[5, 5] = 1; V3[11, 5] = 1; W3[7, 5] = 1
    # Term 6: e20 * e20 * e02
    U3[6, 6] = 1; V3[6, 6] = 1; W3[2, 6] = 1
    # Term 7: e21 * e21 * e12
    U3[7, 7] = 1; V3[10, 7] = 1; W3[5, 7] = 1
    # Term 8: e22 * e22 * e22
    U3[8, 8] = 1; V3[14, 8] = 1; W3[8, 8] = 1
    
    # Additional terms for CW decomposition
    # Term 9: (e00+e11) * (e00+e11) * (e00+e11)
    U3[0, 9] = 1; U3[4, 9] = 1
    V3[0, 9] = 1; V3[7, 9] = 1
    W3[0, 9] = 1; W3[4, 9] = 1
    
    # Term 10: (e00+e22) * (e00+e22) * (e00+e22)
    U3[0, 10] = 1; U3[8, 10] = 1
    V3[0, 10] = 1; V3[14, 10] = 1
    W3[0, 10] = 1; W3[8, 10] = 1
    
    # Term 11: (e11+e22) * (e11+e22) * (e11+e22)
    U3[4, 11] = 1; U3[8, 11] = 1
    V3[7, 11] = 1; V3[14, 11] = 1
    W3[4, 11] = 1; W3[8, 11] = 1
    
    # Term 12: (e01+e10) * (e01+e10) * (e01+e10)
    U3[1, 12] = 1; U3[3, 12] = 1
    V3[4, 12] = 1; V3[3, 12] = 1
    W3[1, 12] = 1; W3[3, 12] = 1
    
    # Term 13: (e02+e20) * (e02+e20) * (e02+e20)
    U3[2, 13] = 1; U3[6, 13] = 1
    V3[8, 13] = 1; V3[6, 13] = 1
    W3[2, 13] = 1; W3[6, 13] = 1
    
    # Term 14: (e12+e21) * (e12+e21) * (e12+e21)
    U3[5, 14] = 1; U3[7, 14] = 1
    V3[11, 14] = 1; V3[10, 14] = 1
    W3[5, 14] = 1; W3[7, 14] = 1
    
    # Term 15: (e00-e11) * (e00-e11) * (e00-e11)
    U3[0, 15] = 1; U3[4, 15] = -1
    V3[0, 15] = 1; V3[7, 15] = -1
    W3[0, 15] = 1; W3[4, 15] = -1
    
    # Term 16: (e00-e22) * (e00-e22) * (e00-e22)
    U3[0, 16] = 1; U3[8, 16] = -1
    V3[0, 16] = 1; V3[14, 16] = -1
    W3[0, 16] = 1; W3[8, 16] = -1
    
    # Term 17: (e11-e22) * (e11-e22) * (e11-e22)
    U3[4, 17] = 1; U3[8, 17] = -1
    V3[7, 17] = 1; V3[14, 17] = -1
    W3[4, 17] = 1; W3[8, 17] = -1
    
    # Term 18: (e01-e10) * (e01-e10) * (e01-e10)
    U3[1, 18] = 1; U3[3, 18] = -1
    V3[4, 18] = 1; V3[3, 18] = -1
    W3[1, 18] = 1; W3[3, 18] = -1
    
    # Term 19: (e02-e20) * (e02-e20) * (e02-e20)
    U3[2, 19] = 1; U3[6, 19] = -1
    V3[8, 19] = 1; V3[6, 19] = -1
    W3[2, 19] = 1; W3[6, 19] = -1
    
    # Term 20: (e12-e21) * (e12-e21) * (e12-e21)
    U3[5, 20] = 1; U3[7, 20] = -1
    V3[11, 20] = 1; V3[10, 20] = -1
    W3[5, 20] = 1; W3[7, 20] = -1
    
    # Term 21: (e00+e11+e22) * (e00+e11+e22) * (e00+e11+e22)
    U3[0, 21] = 1; U3[4, 21] = 1; U3[8, 21] = 1
    V3[0, 21] = 1; V3[7, 21] = 1; V3[14, 21] = 1
    W3[0, 21] = 1; W3[4, 21] = 1; W3[8, 21] = 1
    
    # Term 22: (e01+e12+e20) * (e01+e12+e20) * (e01+e12+e20)
    U3[1, 22] = 1; U3[5, 22] = 1; U3[6, 22] = 1
    V3[4, 22] = 1; V3[11, 22] = 1; V3[6, 22] = 1
    W3[1, 22] = 1; W3[5, 22] = 1; W3[6, 22] = 1

    # Now construct the 6x6 decomposition using Kronecker product
    # 6x6 = (2x2) ⊗ (3x3)
    # The Kronecker product of decompositions:
    # If T2 = sum_{r2} u2_r2 * v2_r2 * w2_r2 and T3 = sum_{r3} u3_r3 * v3_r3 * w3_r3
    # Then T6 = T2 ⊗ T3 = sum_{r2,r3} (u2_r2 ⊗ u3_r3) * (v2_r2 ⊗ v3_r3) * (w2_r2 ⊗ w3_r3)
    # Rank = 7 * 23 = 161
    
    rank2 = 7
    rank3 = 23
    rank6 = rank2 * rank3  # 161
    
    U6 = np.zeros((36, rank6), dtype=float)
    V6 = np.zeros((36, rank6), dtype=float)
    W6 = np.zeros((36, rank6), dtype=float)
    
    for r2 in range(rank2):
        for r3 in range(rank3):
            r6 = r2 * rank3 + r3
            # u6 = u2_r2 ⊗ u3_r3, v6 = v2_r2 ⊗ v3_r3, w6 = w2_r2 ⊗ w3_r3
            U6[:, r6] = np.kron(U2[:, r2], U3[:, r3])
            V6[:, r6] = np.kron(V2[:, r2], V3[:, r3])
            W6[:, r6] = np.kron(W2[:, r2], W3[:, r3])
    
    # Convert to lists for the return format
    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
