import numpy as np

def construct():
    n, m, p = 4, 4, 4
    
    # Strassen 2x2x2 factor matrices (4,7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    # M2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    # M3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    # M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    # M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    r = 49
    U = np.zeros((16, r), dtype=int)
    V = np.zeros((16, r), dtype=int)
    W = np.zeros((16, r), dtype=int)

    for k1 in range(7):
        for k2 in range(7):
            k = k1 * 7 + k2
            U[:, k] = np.kron(U2[:, k1], U2[:, k2])
            V[:, k] = np.kron(V2[:, k1], V2[:, k2])
            W[:, k] = np.kron(W2[:, k1], W2[:, k2])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
