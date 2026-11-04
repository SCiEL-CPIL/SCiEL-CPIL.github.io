import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2x2
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Laderman 3x3x3
    # U3, V3, W3 are 9x7
    U3 = np.zeros((9, 7), dtype=int)
    V3 = np.zeros((9, 7), dtype=int)
    W3 = np.zeros((9, 7), dtype=int)
    
    # Indices: 0..2, 3..5, 6..8
    # M1
    U3[0, 0] = 1; U3[4, 0] = 1; U3[8, 0] = 1
    V3[0, 0] = 1; V3[4, 0] = 1; V3[8, 0] = 1
    W3[0, 0] = 1; W3[4, 0] = 1; W3[8, 0] = 1
    
    # M2
    U3[1, 1] = 1; U3[5, 1] = 1; U3[6, 1] = 1
    V3[1, 1] = 1; V3[5, 1] = 1; V3[6, 1] = 1
    W3[1, 1] = 1; W3[5, 1] = 1; W3[6, 1] = 1
    
    # M3
    U3[2, 2] = 1; U3[3, 2] = 1; U3[7, 2] = 1
    V3[2, 2] = 1; V3[3, 2] = 1; V3[7, 2] = 1
    W3[2, 2] = 1; W3[3, 2] = 1; W3[7, 2] = 1
    
    # M4
    U3[0, 3] = 1; U3[4, 3] = 1; U3[8, 3] = -1
    V3[0, 3] = 1; V3[4, 3] = 1; V3[8, 3] = -1
    W3[0, 3] = 1; W3[4, 3] = 1; W3[8, 3] = -1
    
    # M5
    U3[1, 4] = 1; U3[5, 4] = 1; U3[6, 4] = -1
    V3[1, 4] = 1; V3[5, 4] = 1; V3[6, 4] = -1
    W3[1, 4] = 1; W3[5, 4] = 1; W3[6, 4] = -1
    
    # M6
    U3[2, 5] = 1; U3[3, 5] = 1; U3[7, 5] = -1
    V3[2, 5] = 1; V3[3, 5] = 1; V3[7, 5] = -1
    W3[2, 5] = 1; W3[3, 5] = 1; W3[7, 5] = -1
    
    # M7
    U3[0, 6] = 1; U3[4, 6] = 1; U3[8, 6] = 1
    U3[1, 6] = 1; U3[5, 6] = 1; U3[6, 6] = 1
    U3[2, 6] = 1; U3[3, 6] = 1; U3[7, 6] = 1
    V3[0, 6] = 1; V3[4, 6] = 1; V3[8, 6] = 1
    V3[1, 6] = 1; V3[5, 6] = 1; V3[6, 6] = 1
    V3[2, 6] = 1; V3[3, 6] = 1; V3[7, 6] = 1
    W3[0, 6] = 1; W3[4, 6] = 1; W3[8, 6] = 1
    W3[1, 6] = 1; W3[5, 6] = 1; W3[6, 6] = 1
    W3[2, 6] = 1; W3[3, 6] = 1; W3[7, 6] = 1
    
    # Kronecker Product for 6x6x6
    # U6 = U2 (x) U3
    # Rows: 4*9 = 36
    # Cols: 7*7 = 49
    U6 = np.zeros((36, 49), dtype=int)
    V6 = np.zeros((36, 49), dtype=int)
    W6 = np.zeros((36, 49), dtype=int)
    
    for i2 in range(4):
        for i3 in range(9):
            row6 = i2 * 9 + i3
            for c2 in range(7):
                for c3 in range(7):
                    col6 = c2 * 7 + c3
                    val = U2[i2, c2] * U3[i3, c3]
                    U6[row6, col6] = val
                    V6[row6, col6] = V2[i2, c2] * V3[i3, c3]
                    W6[row6, col6] = W2[i2, c2] * W3[i3, c3]
                    
    # Extract 5x5x5
    # U5, V5: first 25 rows of U6, V6
    U5 = U6[:25, :]
    V5 = V6[:25, :]
    
    # W5: Permute rows of W6
    # Target index t = k*5 + l for k,l in 0..4
    # Source index s = k*6 + l
    W5 = np.zeros((25, 49), dtype=int)
    for k in range(5):
        for l in range(5):
            t = k * 5 + l
            s = k * 6 + l
            W5[t, :] = W6[s, :]
            
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
# EVOLVE-BLOCK-END
