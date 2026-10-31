import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Define 2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for 2x2:
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen formulas for 2x2
    # M1 = (A11 - A00)(B00 + B11)
    # M2 = (A00 + A10)B00
    # M3 = A11(B10 - B00)
    # M4 = A00(B01 - B11)
    # M5 = (A01 + A11)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # C indices: 00->0, 01->1, 10->2, 11->3
    # W index: k*2 + i.
    # C00 (i=0,k=0) -> W[0]
    # C01 (i=0,k=1) -> W[2]
    # C10 (i=1,k=0) -> W[1]
    # C11 (i=1,k=1) -> W[3]
    
    # M1
    U2[3, 0] = 1; U2[0, 0] = -1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[0, 1] = 1; U2[2, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3
    U2[3, 2] = 1
    V2[2, 2] = 1; V2[0, 2] = -1
    W2[2, 2] = 1
    
    # M4
    U2[0, 3] = 1
    V2[1, 3] = 1; V2[3, 3] = -1
    W2[0, 3] = 1; W2[3, 3] = 1
    
    # M5
    U2[1, 4] = 1; U2[3, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = 1; W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct 4x4 factors
    # U4[i*4+j, r] where r = 7*r2 + r2'
    # i = 2*i2 + i2', j = 2*j2 + j2'
    # U4[i*4+j, r] = U2[i2*2+j2, r2] * U2[i2'*2+j2', r2']
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i2 in range(2):
        for j2 in range(2):
            for i2p in range(2):
                for j2p in range(2):
                    # Global A index
                    i = 2 * i2 + i2p
                    j = 2 * j2 + j2p
                    u_idx = i * 4 + j
                    
                    # Local indices for 2x2
                    u2_idx = i2 * 2 + j2
                    u2p_idx = i2p * 2 + j2p
                    
                    for r2 in range(7):
                        for r2p in range(7):
                            r = 7 * r2 + r2p
                            U4[u_idx, r] = U2[u2_idx, r2] * U2[u2p_idx, r2p]
                            
    for j2 in range(2):
        for k2 in range(2):
            for j2p in range(2):
                for k2p in range(2):
                    # Global B index
                    j = 2 * j2 + j2p
                    k = 2 * k2 + k2p
                    v_idx = j * 4 + k
                    
                    # Local indices for 2x2
                    v2_idx = j2 * 2 + k2
                    v2p_idx = j2p * 2 + k2p
                    
                    for r2 in range(7):
                        for r2p in range(7):
                            r = 7 * r2 + r2p
                            V4[v_idx, r] = V2[v2_idx, r2] * V2[v2p_idx, r2p]
                            
    for k2 in range(2):
        for i2 in range(2):
            for k2p in range(2):
                for i2p in range(2):
                    # Global C index for W: k*4 + i
                    k = 2 * k2 + k2p
                    i = 2 * i2 + i2p
                    w_idx = k * 4 + i
                    
                    # Local indices for 2x2
                    # W2 index is k*2 + i
                    w2_idx = k2 * 2 + i2
                    w2p_idx = k2p * 2 + i2p
                    
                    for r2 in range(7):
                        for r2p in range(7):
                            r = 7 * r2 + r2p
                            W4[w_idx, r] = W2[w2_idx, r2] * W2[w2p_idx, r2p]
                            
    # Convert to lists of lists as required by interface? 
    # "integer/half-integer entries"
    # The return type is {'U': ..., 'V': ..., 'W': ...}
    # Usually lists of lists or numpy arrays are accepted if convertible.
    # The prompt says "factor matrices of shapes...".
    # I will return lists of lists to be safe with "integer/half-integer" if float 0.0 is considered float.
    # But numpy arrays are standard. I'll stick to numpy arrays as they are faster and standard.
    
    return {"U": U4, "V": V4, "W": W4}
