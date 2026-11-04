import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 2x2 (Rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1

    # 2. Trivial decomposition for 3x3 (Rank 9)
    # U3: (9, 9), V3: (9, 9), W3: (9, 9)
    # Indices: U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    
    U3 = np.zeros((9, 9), dtype=float)
    V3 = np.zeros((9, 9), dtype=float)
    W3 = np.zeros((9, 9), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j, col] = 1
                V3[j*3 + k, col] = 1
                W3[k*3 + i, col] = 1
                col += 1

    # 3. Kronecker Product to get 6x6 (Rank 63)
    # U6: (36, 63), V6: (36, 63), W6: (36, 63)
    # We construct the 5x5 directly by mapping indices.
    
    R = 7 * 9
    U = np.zeros((25, R), dtype=float)
    V = np.zeros((25, R), dtype=float)
    W = np.zeros((25, R), dtype=float)
    
    # Mapping function for 6x6 to 5x5 indices
    # i_6 = i_2 * 3 + i_3
    # We only care about i_6 < 5.
    
    for i in range(5):
        for j in range(5):
            # Decompose i, j into (i_2, i_3) and (j_2, j_3)
            i_2 = i // 3
            i_3 = i % 3
            j_2 = j // 3
            j_3 = j % 3
            
            idx_u = i * 5 + j
            
            for r2 in range(7):
                for r3 in range(9):
                    r = r2 * 9 + r3
                    val_u = U2[i_2 * 2 + j_2, r2] * U3[i_3 * 3 + j_3, r3]
                    if val_u != 0:
                        U[idx_u, r] = val_u
                        
    for j in range(5):
        for k in range(5):
            j_2 = j // 3
            j_3 = j % 3
            k_2 = k // 3
            k_3 = k % 3
            
            idx_v = j * 5 + k
            
            for r2 in range(7):
                for r3 in range(9):
                    r = r2 * 9 + r3
                    val_v = V2[j_2 * 2 + k_2, r2] * V3[j_3 * 3 + k_3, r3]
                    if val_v != 0:
                        V[idx_v, r] = val_v

    for k in range(5):
        for i in range(5):
            k_2 = k // 3
            k_3 = k % 3
            i_2 = i // 3
            i_3 = i % 3
            
            idx_w = k * 5 + i
            
            for r2 in range(7):
                for r3 in range(9):
                    r = r2 * 9 + r3
                    val_w = W2[k_2 * 2 + i_2, r2] * W3[k_3 * 3 + i_3, r3]
                    if val_w != 0:
                        W[idx_w, r] = val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
