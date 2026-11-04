import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Rank-7 Strassen decomposition for 2x2x2
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Construct 4x4x4 decomposition (Rank 49)
    # We need to permute the rows of the Kronecker products to match i*m+j order.
    # Permutation for 4x4 indices from Kronecker order
    # Kronecker index k = r1*4 + r2
    # r1 = i1*2 + j1, r2 = i2*2 + j2
    # k = i1*8 + j1*4 + i2*2 + j2
    # Target index t = i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # We want U4[t, c] = (U2[k1, r1] * U2[k2, r2]) where k1, k2 are derived from t.
    # Actually, it's easier to just build U4 by iterating over columns.
    
    rank4 = 49
    U4 = np.zeros((16, rank4))
    V4 = np.zeros((16, rank4))
    W4 = np.zeros((16, rank4))
    
    # Mapping from target row index t to Kronecker components
    # t = i*4 + j
    # i = i1*2 + i2
    # j = j1*2 + j2
    # We need to fill U4[t, c] for c = r1*7 + r2
    # U4[t, c] = U2[k1, r1] * U2[k2, r2]
    # where k1 is the row index in U2 corresponding to (i1, j1)
    # k2 is the row index in U2 corresponding to (i2, j2)
    # k1 = i1*2 + j1
    # k2 = i2*2 + j2
    
    for t in range(16):
        i = t // 4
        j = t % 4
        i1 = i // 2
        i2 = i % 2
        j1 = j // 2
        j2 = j % 2
        k1 = i1 * 2 + j1
        k2 = i2 * 2 + j2
        
        for r1 in range(7):
            for r2 in range(7):
                c = r1 * 7 + r2
                U4[t, c] = U2[k1, r1] * U2[k2, r2]
                V4[t, c] = V2[k1, r1] * V2[k2, r2]
                W4[t, c] = W2[k1, r1] * W2[k2, r2]
                
    # Now construct 4x4x8 decomposition (Rank 98)
    rank8 = 98
    U8 = np.zeros((16, rank8))
    V8 = np.zeros((32, rank8))
    W8 = np.zeros((32, rank8))
    
    # First 49 columns
    U8[:, 0:49] = U4
    V8[0:16, 0:49] = V4
    W8[0:16, 0:49] = W4
    
    # Second 49 columns
    U8[:, 49:98] = U4
    V8[16:32, 49:98] = V4
    W8[16:32, 49:98] = W4
    
    # Convert to lists of lists for the return format
    U = U8.tolist()
    V = V8.tolist()
    W = W8.tolist()
    
    return {"U": U, "V": V, "W": W}
