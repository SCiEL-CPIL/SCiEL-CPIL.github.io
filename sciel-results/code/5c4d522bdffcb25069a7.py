import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # 2x2x2 Winograd decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # Term 2: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # Term 3: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1
    
    # Term 4: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[3, 6] = 1

    # Construct 4x4x4 decomposition (Rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for a in range(7):
        for b in range(7):
            col = a * 7 + b
            for i_o in range(2):
                for j_o in range(2):
                    for k_o in range(2):
                        u_o = U2[i_o * 2 + j_o, a]
                        v_o = V2[j_o * 2 + k_o, a]
                        w_o = W2[k_o * 2 + i_o, a]
                        
                        for i_i in range(2):
                            for j_i in range(2):
                                for k_i in range(2):
                                    u_i = U2[i_i * 2 + j_i, b]
                                    v_i = V2[j_i * 2 + k_i, b]
                                    w_i = W2[k_i * 2 + i_i, b]
                                    
                                    i = i_o * 2 + i_i
                                    j = j_o * 2 + j_i
                                    k = k_o * 2 + k_i
                                    
                                    U4[i * 4 + j, col] = u_o * u_i
                                    V4[j * 4 + k, col] = v_o * v_i
                                    W4[k * 4 + i, col] = w_o * w_i

    # Extend to 4x4x9
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Block 1: k=0..3
    U[:, 0:49] = U4
    for r in range(16):
        j = r // 4
        k = r % 4
        V[j * 9 + k, 0:49] = V4[r, :]
    for r in range(16):
        k = r // 4
        i = r % 4
        W[k * 4 + i, 0:49] = W4[r, :]
        
    # Block 2: k=4..7
    U[:, 49:98] = U4
    for r in range(16):
        j = r // 4
        k = r % 4
        V[j * 9 + (k + 4), 49:98] = V4[r, :]
    for r in range(16):
        k = r // 4
        i = r % 4
        W[(k + 4) * 4 + i, 49:98] = W4[r, :]
        
    # Block 3: k=8
    for i in range(4):
        for j in range(4):
            col = 98 + (i * 4 + j)
            U[i * 4 + j, col] = 1
            V[j * 9 + 8, col] = 1
            W[8 * 4 + i, col] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
