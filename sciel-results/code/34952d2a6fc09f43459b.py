import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Strassen 2x2 factors
    # U2[i*2+j, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 1] = -1; W2[0, 6] = -1
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[0, 1] = -1
    # M3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[0, 2] = 1
    # M4
    U2[3, 3] = 1
    V2[0, 3] = 1; V2[1, 3] = -1
    W2[0, 3] = 1
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1; W2[0, 6] = -1
    
    # Fill U, V, W
    for r1 in range(7):
        for r2 in range(7):
            c = r1 * 7 + r2
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    ip = i % 2
                    J = j // 2
                    jp = j % 2
                    U[i * 4 + j, c] = U2[I * 2 + J, r1] * U2[ip * 2 + jp, r2]
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    jp = j % 2
                    K = k // 2
                    kp = k % 2
                    V[j * 4 + k, c] = V2[J * 2 + K, r1] * V2[jp * 2 + kp, r2]
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    kp = k % 2
                    I = i // 2
                    ip = i % 2
                    W[k * 4 + i, c] = W2[K * 2 + I, r1] * W2[kp * 2 + ip, r2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
