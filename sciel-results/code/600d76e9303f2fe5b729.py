import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # U1, V1, W1 are 4x7 matrices
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U1 = np.array([
        [ 1,  0,  1,  0,  1,  1,  0],   # A00
        [ 0,  0,  0,  0,  1,  0,  1],   # A01
        [ 0,  1,  0,  0,  0, -1,  0],   # A10
        [ 1,  1,  0,  1,  0,  0, -1],   # A11
    ], dtype=np.int64)

    V1 = np.array([
        [ 1,  1,  0,  0,  0,  1,  0],   # B00
        [ 0,  0,  1,  0,  0,  1,  0],   # B01
        [ 0,  0,  0,  1,  0,  0,  1],   # B10
        [ 1,  0, -1, -1,  1,  0,  1],   # B11
    ], dtype=np.int64)

    W1 = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],   # C00
        [ 0,  0,  1,  0,  1,  0,  0],   # C01
        [ 0,  1,  0,  1,  0,  0,  0],   # C10
        [ 1, -1,  1,  0,  0,  1,  0],   # C11
    ], dtype=np.int64)

    # Trivial decomposition for 1x2x3 (rank 6)
    # A2 is 1x2: indices 0=(0,0), 1=(0,1)
    # B2 is 2x3: indices 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    # C2 is 1x3: indices 0=(0,0), 1=(0,1), 2=(0,2)
    
    # Rank 1 terms:
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    # c2 = a0*b02 + a1*b12
    
    # U2 (2x6): rows a0, a1
    U2 = np.zeros((2, 6), dtype=np.int64)
    U2[0, 0] = 1 # a0 * b00
    U2[1, 3] = 1 # a1 * b10
    U2[0, 1] = 1 # a0 * b01
    U2[1, 4] = 1 # a1 * b11
    U2[0, 2] = 1 # a0 * b02
    U2[1, 5] = 1 # a1 * b12
    
    # V2 (6x6): rows b00, b01, b02, b10, b11, b12
    V2 = np.zeros((6, 6), dtype=np.int64)
    V2[0, 0] = 1 # b00
    V2[3, 3] = 1 # b10
    V2[1, 1] = 1 # b01
    V2[4, 4] = 1 # b11
    V2[2, 2] = 1 # b02
    V2[5, 5] = 1 # b12
    
    # W2 (3x6): rows c0, c1, c2
    W2 = np.zeros((3, 6), dtype=np.int64)
    W2[0, 0] = 1 # c0 from a0*b00
    W2[0, 3] = 1 # c0 from a1*b10
    W2[1, 1] = 1 # c1 from a0*b01
    W2[1, 4] = 1 # c1 from a1*b11
    W2[2, 2] = 1 # c2 from a0*b02
    W2[2, 5] = 1 # c2 from a1*b12

    # Kronecker products
    U_kron = np.kron(U1, U2) # 8x42
    V_kron = np.kron(V1, V2) # 24x42
    W_kron = np.kron(W1, W2) # 12x42
    
    # Permute V_kron rows to match global B index order
    # Global B index: j * 6 + k
    # j = j1 * 2 + j2, k = k1 * 3 + k2
    # Global index = (j1 * 2 + j2) * 6 + (k1 * 3 + k2) = 12 j1 + 6 j2 + 3 k1 + k2
    
    # Kron V index: v1 * 6 + v2
    # v1 = j1 * 2 + k1 (from V1 index)
    # v2 = j2 * 3 + k2 (from V2 index)
    # Kron index = (j1 * 2 + k1) * 6 + (j2 * 3 + k2) = 12 j1 + 6 k1 + 3 j2 + k2
    
    # We need to map Kron Index -> Global Index
    # Create a permutation array perm such that V_new[perm[i]] = V_kron[i]
    # Or V_new[global_idx] = V_kron[kron_idx]
    
    V_new = np.zeros_like(V_kron)
    
    # Iterate over all global indices
    for j in range(4):
        for k in range(6):
            global_idx = j * 6 + k
            j1 = j // 2
            j2 = j % 2
            k1 = k // 3
            k2 = k % 3
            
            # Calculate corresponding Kron index
            v1 = j1 * 2 + k1
            v2 = j2 * 3 + k2
            kron_idx = v1 * 6 + v2
            
            V_new[global_idx] = V_kron[kron_idx]
            
    # Verify W index mapping
    # Global C index: i * 6 + k
    # i = i1, k = k1 * 3 + k2
    # Global = 6 i1 + 3 k1 + k2
    
    # Kron W index: w1 * 3 + w2
    # w1 = i1 * 2 + k1 (from W1 index)
    # w2 = k2 (from W2 index, i2=0)
    # Kron = (i1 * 2 + k1) * 3 + k2 = 6 i1 + 3 k1 + k2
    # Matches! No permutation needed for W.
    
    # Verify U index mapping
    # Global A index: i * 4 + j
    # i = i1, j = j1 * 2 + j2
    # Global = 4 i1 + 2 j1 + j2
    
    # Kron U index: u1 * 2 + u2
    # u1 = i1 * 2 + j1 (from U1 index)
    # u2 = j2 (from U2 index, i2=0)
    # Kron = (i1 * 2 + j1) * 2 + j2 = 4 i1 + 2 j1 + j2
    # Matches! No permutation needed for U.

    if time.time() - start_time > 110:
        return {"U": U_kron.tolist(), "V": V_new.tolist(), "W": W_kron.tolist()}

    return {"U": U_kron.tolist(), "V": V_new.tolist(), "W": W_kron.tolist()}
