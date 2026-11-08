import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 49 decomposition via tensor product of two rank 7 Strassen decompositions
    n, m, p = DIMENSIONS
    
    # Rank 7 Strassen decomposition for 2x2 matrices
    # Indices for 2x2: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Rows of U correspond to A entries, V to B entries, W to C entries.
    # Columns correspond to the 7 products.
    
    # Strassen products:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B11)
    # P4 = A11(B01 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # P1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00 gets P1, C11 gets P1?
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's fill W2 based on C formulas
    # C00 (index 0): P1, P4, -P5, P7
    W2[0, 0] = 1
    W2[0, 3] = 1
    W2[0, 4] = -1
    W2[0, 6] = 1
    
    # C01 (index 1): P3, P5
    W2[1, 2] = 1
    W2[1, 4] = 1
    
    # C10 (index 2): P2, P4
    W2[2, 1] = 1
    W2[2, 3] = 1
    
    # C11 (index 3): P1, -P2, P3, P6
    W2[3, 0] = 1
    W2[3, 1] = -1
    W2[3, 2] = 1
    W2[3, 5] = 1
    
    # Now fill U2 and V2
    # P1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    
    # P2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    
    # P3: A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    
    # P4: A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    
    # P5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    
    # P6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    
    # P7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # Construct 4x4 decomposition
    # Rank = 7 * 7 = 49
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Map 4x4 index (i, j) to 0..3
    # i = 2*i1 + i2, j = 2*j1 + j2
    # For each pair of Strassen products (c1, c2) from 0..6:
    # The term contributes to U, V, W with indices derived from Kronecker product
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            # U index: (a1, a2) where a1 is from U2[:, c1] and a2 is from U2[:, c2]?
            # Wait, the tensor product structure:
            # A_{(i1, i2), (k1, k2)} = A_{i1, k1}^{(1)} A_{i2, k2}^{(2)}? No.
            # The tensor T_{i, j, k} = sum T^{(1)}_{i1, j1, k1} T^{(2)}_{i2, j2, k2}
            # where i = 2*i1 + i2, etc.
            # So U_{i, c} = U2_{i1, c1} * U2_{i2, c2}
            # V_{j, c} = V2_{j1, c1} * V2_{j2, c2}
            # W_{k, c} = W2_{k1, c1} * W2_{k2, c2}
            
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    val_u = U2[i1, c1] * U2[i2, c2]
                    if val_u != 0:
                        U[i, col] = val_u
            
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    val_v = V2[j1, c1] * V2[j2, c2]
                    if val_v != 0:
                        V[j, col] = val_v
            
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    val_w = W2[k1, c1] * W2[k2, c2]
                    if val_w != 0:
                        W[k, col] = val_w
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
