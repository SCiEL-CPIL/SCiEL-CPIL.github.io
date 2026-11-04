"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    # Strassen decomposition for 2x2 matrices
    # Indices for U2: i*2+j
    # Indices for V2: j*2+k
    # Indices for W2: k*2+i
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: M2 = (A10 + A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # Term 3: M3 = A00(B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # Term 4: M4 = A11(B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # Term 5: M5 = (A00 + A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # Term 6: M6 = (A10 - A00)(B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: M7 = (A01 - A11)(B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Construct 4x4 decomposition using Kronecker product structure
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            rU = i * 4 + j
                            U4[rU, c] += U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = 2 * k1 + k2
                                    rV = j * 4 + k
                                    V4[rV, c] += V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
                                    
                                    rW = k * 4 + i
                                    W4[rW, c] += W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END
