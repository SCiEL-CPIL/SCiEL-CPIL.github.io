import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    # Rows of U correspond to (0,0), (0,1), (1,0), (1,1)
    # Rows of V correspond to (0,0), (0,1), (1,0), (1,1)
    # Rows of W correspond to (0,0), (0,1), (1,0), (1,1)
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct 4x4x4 decomposition (Rank 49)
    # U4, V4, W4 are 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Map 4x4 index (i, j) to outer (I, J) and inner (ip, jp)
    # i = 2*I + ip, j = 2*J + jp
    # U2 index for outer: 2*I + J
    # U2 index for inner: 2*ip + jp
    
    for i in range(4):
        for j in range(4):
            I = i // 2
            J = j // 2
            ip = i % 2
            jp = j % 2
            idx4 = i * 4 + j
            idx2_outer = 2 * I + J
            idx2_inner = 2 * ip + jp
            
            for ro in range(7):
                uo = U2[idx2_outer, ro]
                if uo == 0:
                    continue
                for ri in range(7):
                    ui = U2[idx2_inner, ri]
                    if ui == 0:
                        continue
                    r = ro * 7 + ri
                    U4[idx4, r] = uo * ui
                    
            # V index: j * 4 + k
            # j = 2*J + jp, k = 2*K + kp
            # V2 index outer: 2*J + K
            # V2 index inner: 2*jp + kp
            for k in range(4):
                K = k // 2
                kp = k % 2
                idx4_v = j * 4 + k
                idx2_v_outer = 2 * J + K
                idx2_v_inner = 2 * jp + kp
                
                for ro in range(7):
                    vo = V2[idx2_v_outer, ro]
                    if vo == 0:
                        continue
                    for ri in range(7):
                        vi = V2[idx2_v_inner, ri]
                        if vi == 0:
                            continue
                        r = ro * 7 + ri
                        V4[idx4_v, r] = vo * vi
                        
            # W index: k * 4 + i
            # k = 2*K + kp, i = 2*I + ip
            # W2 index outer: 2*K + I
            # W2 index inner: 2*kp + ip
            for k in range(4):
                K = k // 2
                kp = k % 2
                idx4_w = k * 4 + i
                idx2_w_outer = 2 * K + I
                idx2_w_inner = 2 * kp + ip
                
                for ro in range(7):
                    wo = W2[idx2_w_outer, ro]
                    if wo == 0:
                        continue
                    for ri in range(7):
                        wi = W2[idx2_w_inner, ri]
                        if wi == 0:
                            continue
                        r = ro * 7 + ri
                        W4[idx4_w, r] = wo * wi

    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Right block (cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
