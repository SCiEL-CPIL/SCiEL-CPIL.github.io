import numpy as np

def construct():
    n = 4
    rank = 49
    
    # Strassen 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Build 4x4 via block Kronecker
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                for K in range(7):
                    ub = U2[ib*2+jb, K]
                    vb = V2[jb*2+kb, K]
                    wb = W2[ib*2+kb, K]
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                I = ib*2 + i
                                J = jb*2 + j
                                Kg = kb*2 + k
                                uidx = I*4 + J
                                vidx = J*4 + Kg
                                widx = Kg*4 + I
                                for kp in range(7):
                                    col = K*7 + kp
                                    U[uidx, col] += ub * U2[i*2+j, kp]
                                    V[vidx, col] += vb * V2[j*2+k, kp]
                                    W[widx, col] += wb * W2[i*2+k, kp]
    
    return {"U": U, "V": V, "W": W}
