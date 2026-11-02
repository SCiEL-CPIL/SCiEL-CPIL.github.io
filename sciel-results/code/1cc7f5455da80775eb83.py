import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    # 1. Define Strassen factors for 2x2x2
    # U1, V1, W1 shapes (4, 7)
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # r=0: M1
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # r=1: M2
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1; W1[3, 1] = 1
    
    # r=2: M3
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    
    # r=3: M4
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # r=4: M5
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # r=5: M6
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # r=6: M7
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    # 2. Define Schoolbook factors for 2x2x3
    # U2, V2, W2 shapes (4, 12), (6, 12), (6, 12)
    U2 = np.zeros((4, 12))
    V2 = np.zeros((6, 12))
    W2 = np.zeros((6, 12))
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U2[i * 2 + j, col] = 1
                V2[j * 3 + k, col] = 1
                W2[k * 2 + i, col] = 1
                col += 1

    # 3. Construct Final Factors
    rank1 = 7
    rank2 = 12
    rank = rank1 * rank2
    
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Pre-compute non-zero indices for efficiency, though loops are fine for this size
    # U1 non-zeros: (idx, val)
    # V1 non-zeros
    # W1 non-zeros
    # Same for 2
    
    nz_U1 = [(idx, val) for idx in range(4) for r in range(7) if U1[idx, r] != 0 for val in [U1[idx, r]]]
    # Better: list of (r, idx, val)
    nz_U1 = []
    for r in range(7):
        for idx in range(4):
            if U1[idx, r] != 0:
                nz_U1.append((r, idx, U1[idx, r]))
                
    nz_V1 = []
    for r in range(7):
        for idx in range(4):
            if V1[idx, r] != 0:
                nz_V1.append((r, idx, V1[idx, r]))
                
    nz_W1 = []
    for r in range(7):
        for idx in range(4):
            if W1[idx, r] != 0:
                nz_W1.append((r, idx, W1[idx, r]))
                
    nz_U2 = []
    for r in range(12):
        for idx in range(4):
            if U2[idx, r] != 0:
                nz_U2.append((r, idx, U2[idx, r]))
                
    nz_V2 = []
    for r in range(12):
        for idx in range(6):
            if V2[idx, r] != 0:
                nz_V2.append((r, idx, V2[idx, r]))
                
    nz_W2 = []
    for r in range(12):
        for idx in range(6):
            if W2[idx, r] != 0:
                nz_W2.append((r, idx, W2[idx, r]))

    # Iterate over all rank combinations
    for r1 in range(7):
        for r2 in range(12):
            r = r1 * 12 + r2
            
            # U
            # idx_U1 = i1 * 2 + j1
            # idx_U2 = i2 * 2 + j2
            # idx_U = (2*i1 + i2)*4 + (2*j1 + j2)
            u1s = [item for item in nz_U1 if item[0] == r1]
            u2s = [item for item in nz_U2 if item[0] == r2]
            for _, idx1, val1 in u1s:
                i1 = idx1 // 2
                j1 = idx1 % 2
                for _, idx2, val2 in u2s:
                    i2 = idx2 // 2
                    j2 = idx2 % 2
                    idx = (2 * i1 + i2) * 4 + (2 * j1 + j2)
                    U[idx, r] += val1 * val2
                    
            # V
            # idx_V1 = j1 * 2 + k1
            # idx_V2 = j2 * 3 + k2
            # idx_V = (2*j1 + j2)*6 + (3*k1 + k2)
            v1s = [item for item in nz_V1 if item[0] == r1]
            v2s = [item for item in nz_V2 if item[0] == r2]
            for _, idx1, val1 in v1s:
                j1 = idx1 // 2
                k1 = idx1 % 2
                for _, idx2, val2 in v2s:
                    j2 = idx2 // 3
                    k2 = idx2 % 3
                    idx = (2 * j1 + j2) * 6 + (3 * k1 + k2)
                    V[idx, r] += val1 * val2
                    
            # W
            # idx_W1 = k1 * 2 + i1
            # idx_W2 = k2 * 2 + i2
            # idx_W = (3*k1 + k2)*4 + (2*i1 + i2)
            w1s = [item for item in nz_W1 if item[0] == r1]
            w2s = [item for item in nz_W2 if item[0] == r2]
            for _, idx1, val1 in w1s:
                k1 = idx1 // 2
                i1 = idx1 % 2
                for _, idx2, val2 in w2s:
                    k2 = idx2 // 2
                    i2 = idx2 % 2
                    idx = (3 * k1 + k2) * 4 + (2 * i1 + i2)
                    W[idx, r] += val1 * val2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
