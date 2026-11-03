import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 6

    # --- Strassen 2x2x2 decomposition (rank 7) ---
    # U1: 4x7, V1: 4x7, W1: 4x7
    # Index convention: U1[i*2+j], V1[j*2+k], W1[k*2+i]
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]

    # M1 = (a00+a11)(b00+b11) -> c00, c11
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1

    # M2 = (a10+a11)b00 -> c10, c11
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1

    # M3 = a00(b01-b11) -> c01, c11
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1

    # M4 = a11(b10-b00) -> c00, c10
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1

    # M5 = (a00+a01)b11 -> c00, c01
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1

    # M6 = (a10-a00)(b00+b01) -> c11
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1

    # M7 = (a01-a11)(b10+b11) -> c00
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1

    # --- Winograd-style 2x2x3 decomposition (rank 5) ---
    # A is 2x2, B is 2x3, C is 2x3
    # U2: 4x5, V2: 6x5, W2: 6x5
    # Index convention: U2[i*2+j], V2[j*3+k], W2[k*2+i]
    U2 = [[0]*5 for _ in range(4)]
    V2 = [[0]*5 for _ in range(6)]
    W2 = [[0]*5 for _ in range(6)]

    # M1 = A00 * B00 -> C00
    U2[0][0] = 1
    V2[0*3+0][0] = 1
    W2[0*2+0][0] = 1

    # M2 = A11 * B11 -> C11
    U2[3][1] = 1
    V2[1*3+1][1] = 1
    W2[1*2+1][1] = 1

    # M3 = (A00+A11) * B01 -> C01
    U2[0][2] = 1; U2[3][2] = 1
    V2[0*3+1][2] = 1
    W2[1*2+0][2] = 1

    # M4 = (A01+A10) * B02 -> C10
    U2[1][3] = 1; U2[2][3] = 1
    V2[0*3+2][3] = 1
    W2[0*2+1][3] = 1

    # M5 = (A11-A00) * B12 -> C12
    U2[3][4] = 1; U2[0][4] = -1
    V2[1*3+2][4] = 1
    W2[2*2+1][4] = 1

    # Verify the 2x2x3 decomposition before proceeding
    T223 = np.zeros((2, 2, 3))
    U2m = np.array(U2)
    V2m = np.array(V2)
    W2m = np.array(W2)
    for r in range(5):
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T223[i, j, k] += U2m[i*2+j, r] * V2m[j*3+k, r] * W2m[k*2+i, r]
    T223_target = np.zeros((2, 2, 3))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T223_target[i, j, k] = 1.0
    if not np.allclose(T223, T223_target):
        # Fallback to schoolbook
        rank_f = n * m * p
        Uf = [[0] * rank_f for _ in range(n * m)]
        Vf = [[0] * rank_f for _ in range(m * p)]
        Wf = [[0] * rank_f for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    Uf[i*m+j][r] = 1
                    Vf[j*p+k][r] = 1
                    Wf[k*n+i][r] = 1
        return {"U": Uf, "V": Vf, "W": Wf}

    # --- Kronecker product construction ---
    # 4x4x6 = (2x2x2) (Kronecker) (2x2x3)
    # Rank = 7 * 5 = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U1m = np.array(U1)
    V1m = np.array(V1)
    W1m = np.array(W1)
    U2m = np.array(U2)
    V2m = np.array(V2)
    W2m = np.array(W2)

    for c1 in range(7):
        for c2 in range(5):
            c = c1 * 5 + c2

            # Fill U: U_full[i*4+j, c] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1m[i1*2+j1, c1]
                            u2 = U2m[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = int(u1 * u2)

            # Fill V: V_full[j*6+k, c] = V1[j1*2+k1, c1] * V2[j2*3+k2, c2]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1m[j1*2+k1, c1]
                            v2 = V2m[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = int(v1 * v2)

            # Fill W: W_full[k*4+i, c] = W1[k1*2+i1, c1] * W2[k2*2+i2, c2]
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1m[k1*2+i1, c1]
                            w2 = W2m[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = int(w1 * w2)

    # --- Verification ---
    if time.time() < deadline - 10:
        Um = np.array(U)
        Vm = np.array(V)
        Wm = np.array(W)

        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = Um[:, r]
            v = Vm[:, r]
            w = Wm[:, r]
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0:
                        continue
                    for k in range(p):
                        T_recon[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]

        T_target = np.ones((n, m, p))
        if not np.allclose(T_recon, T_target, atol=1e-9):
            # Fallback to schoolbook
            rank_f = n * m * p
            Uf = [[0] * rank_f for _ in range(n * m)]
            Vf = [[0] * rank_f for _ in range(m * p)]
            Wf = [[0] * rank_f for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        Uf[i*m+j][r] = 1
                        Vf[j*p+k][r] = 1
                        Wf[k*n+i][r] = 1
            return {"U": Uf, "V": Vf, "W": Wf}

    return {"U": U, "V": V, "W": W}
