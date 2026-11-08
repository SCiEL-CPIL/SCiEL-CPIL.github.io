import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    # Strassen coefficients for 2x2 x 2x2
    # A-blocks: 0=00, 1=01, 2=10, 3=11
    # B-blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        [1, 0, 0, 1],  # A00 + A11
        [0, 0, 1, 1],  # A10 + A11
        [1, 0, 0, 0],  # A00
        [0, 0, 0, 1],  # A11
        [1, 1, 0, 0],  # A00 + A01
        [-1, 0, 1, 0], # A10 - A00
        [0, 1, 0, -1], # A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # B00 + B11
        [1, 0, 0, 0],  # B00
        [0, 1, 0, -1], # B01 - B11
        [-1, 0, 1, 0], # B10 - B00
        [0, 0, 0, 1],  # B11
        [1, 1, 0, 0],  # B00 + B01
        [0, 0, 1, 1],  # B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)],                # s=6
    ]

    # Build Strassen matrices for 2x2 x 2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    for s in range(7):
        for a in range(4):
            if alpha[s][a] != 0:
                U2[a, s] = alpha[s][a]
        for b in range(4):
            if beta[s][b] != 0:
                V2[b, s] = beta[s][b]
        for a, b, gamma in recon[s]:
            W2[a * 2 + b, s] = gamma

    # Kronecker product of factor matrices
    # U: (n*m, R) = kron(U2, U2) -> (16, 49)
    # V: (m*p, R) = kron(V2, V2) -> (24, 49)
    # W: (p*n, R) = kron(W2, W2) -> (24, 49)
    U = np.kron(U2, U2)  # (16, 49)
    V = np.kron(V2, V2)  # (24, 49)
    W = np.kron(W2, W2)  # (24, 49)

    R = 49

    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    r.append(int(rx))
                else:
                    r.append(float(x))
            result.append(r)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
