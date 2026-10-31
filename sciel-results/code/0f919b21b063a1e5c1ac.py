import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS
    rank = 49

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2 coefficients: beta[s, I, J], gamma[s, J, K], delta[s, I, K]
    # s = 0..6, I,J,K = 0..1
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))

    # s=0: M1 = (A11+A22)*(B11+B22)
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    # s=1: M2 = (A21+A22)*B11
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    # s=2: M3 = A11*(B12-B22)
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    # s=3: M4 = A22*(B21-B11)
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    # s=4: M5 = (A11+A12)*B22
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    # s=5: M6 = (A21-A11)*(B11+B12)
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    # s=6: M7 = (A12-A22)*(B21+B22)
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    # 2x2 Strassen U2, V2, W2 (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # t=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # t=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # t=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # t=3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # t=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # t=5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # t=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build U, V, W via recursive Kronecker structure
    for s in range(7):
        for t in range(7):
            c = 7 * s + t

            # U: row = i*4+j, where i=2*I+i', j=2*J+j'
            # U[i*4+j, c] = beta[s, I, J] * U2[i'*2+j', t]
            for i in range(4):
                for j in range(4):
                    I = i >> 1
                    ip = i & 1
                    J = j >> 1
                    jp = j & 1
                    val = beta[s, I, J] * U2[ip * 2 + jp, t]
                    if val != 0:
                        U[i * 4 + j, c] = val

            # V: row = j*4+k, where j=2*J+j', k=2*K+k'
            # V[j*4+k, c] = gamma[s, J, K] * V2[j'*2+k', t]
            for j in range(4):
                for k in range(4):
                    J = j >> 1
                    jp = j & 1
                    K = k >> 1
                    kp = k & 1
                    val = gamma[s, J, K] * V2[jp * 2 + kp, t]
                    if val != 0:
                        V[j * 4 + k, c] = val

            # W: row = k*4+i, where k=2*K+k', i=2*I+i'
            # W[k*4+i, c] = delta[s, I, K] * W2[k'*2+i', t]
            for k in range(4):
                for i in range(4):
                    K = k >> 1
                    kp = k & 1
                    I = i >> 1
                    ip = i & 1
                    val = delta[s, I, K] * W2[kp * 2 + ip, t]
                    if val != 0:
                        W[k * 4 + i, c] = val

            if time.time() > deadline:
                break

        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
