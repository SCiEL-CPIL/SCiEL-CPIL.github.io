import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Winograd 3x3 decomposition (Rank 8)
    U3 = np.zeros((9, 8), dtype=int)
    V3 = np.zeros((9, 8), dtype=int)
    W3 = np.zeros((9, 8), dtype=int)
    
    # M1
    U3[0, 0] = 1; U3[3, 0] = 1; U3[6, 0] = 1
    V3[0, 0] = 1; V3[3, 0] = 1; V3[6, 0] = 1
    W3[0, 0] = 1; W3[3, 0] = 1; W3[6, 0] = 1
    
    # M2
    U3[0, 1] = 1; U3[3, 1] = -1
    V3[0, 1] = 1; V3[3, 1] = -1
    W3[0, 1] = 1; W3[3, 1] = 1
    
    # M3
    U3[1, 2] = 1; U3[7, 2] = 1
    V3[1, 2] = 1; V3[7, 2] = 1
    W3[1, 2] = 1; W3[7, 2] = 1
    
    # M4
    U3[1, 3] = 1; U3[7, 3] = -1
    V3[1, 3] = 1; V3[7, 3] = -1
    W3[1, 3] = 1; W3[7, 3] = 1
    
    # M5
    U3[5, 4] = 1; U3[8, 4] = 1
    V3[5, 4] = 1; V3[8, 4] = 1
    W3[5, 4] = 1; W3[8, 4] = 1
    
    # M6
    U3[5, 5] = 1; U3[8, 5] = -1
    V3[5, 5] = 1; V3[8, 5] = -1
    W3[5, 5] = 1; W3[8, 5] = 1
    
    # M7
    U3[2, 6] = 1; U3[5, 6] = 1
    V3[2, 6] = 1; V3[5, 6] = 1
    W3[2, 6] = 1; W3[5, 6] = 1
    
    # M8
    U3[4, 7] = 1; U3[8, 7] = 1
    V3[4, 7] = 1; V3[8, 7] = 1
    W3[4, 7] = 1; W3[8, 7] = 1

    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    rank = 8 * 7
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    nz_u2 = {r: np.nonzero(U2[:, r])[0] for r in range(7)}
    nz_v2 = {r: np.nonzero(V2[:, r])[0] for r in range(7)}
    nz_w2 = {r: np.nonzero(W2[:, r])[0] for r in range(7)}
    nz_u3 = {r: np.nonzero(U3[:, r])[0] for r in range(8)}
    nz_v3 = {r: np.nonzero(V3[:, r])[0] for r in range(8)}
    nz_w3 = {r: np.nonzero(W3[:, r])[0] for r in range(8)}
    
    for r2 in range(7):
        for r3 in range(8):
            r = r2 * 8 + r3
            
            for u2 in nz_u2[r2]:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3[r3]:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    U6[g_u, r] = val_u2 * val_u3

            for v2 in nz_v2[r2]:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3[r3]:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    V6[g_v, r] = val_v2 * val_v3

            for w2 in nz_w2[r2]:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3[r3]:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    W6[g_w, r] = val_w2 * val_w3

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
