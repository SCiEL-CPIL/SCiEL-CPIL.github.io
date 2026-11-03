"""Matrix multiplication 4,4,9 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 9

    # Build Strassen 2x2x2 (rank 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # M1: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    # M2: A11(B10-B00)
    U2[3][1] = 1
    V2[2][1] = 1; V2[0][1] = -1
    W2[1][1] = 1

    # M3: (A10+A11)B01
    U2[2][2] = 1; U2[3][2] = 1
    V2[1][2] = 1
    W2[2][2] = 1

    # M4: A00(B01-B11)
    U2[0][3] = 1
    V2[1][3] = 1; V2[3][3] = -1
    W2[0][3] = 1

    # M5: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1

    # M6: (A10-A00)(B00+B10)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[0][5] = 1; W2[1][5] = 1

    # M7: (A01-A11)(B01+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[1][6] = 1; V2[3][6] = 1
    W2[2][6] = 1; W2[3][6] = 1

    # Kronecker product to get 4x4x4 (rank 49)
    r4 = 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]

    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                u1 = U2[i1*2+j1][c1]
                if u1 == 0: continue
                for i2 in range(2):
                    for j2 in range(2):
                        for c2 in range(7):
                            u2 = U2[i2*2+j2][c2]
                            if u2 == 0: continue
                            idx_u = (i1*2 + i2)*4 + (j1*2 + j2)
                            idx_c = c1*7 + c2
                            U4[idx_u][idx_c] = u1 * u2

    for j1 in range(2):
        for k1 in range(2):
            for c1 in range(7):
                v1 = V2[j1*2+k1][c1]
                if v1 == 0: continue
                for j2 in range(2):
                    for k2 in range(2):
                        for c2 in range(7):
                            v2 = V2[j2*2+k2][c2]
                            if v2 == 0: continue
                            idx_v = (j1*2 + j2)*4 + (k1*2 + k2)
                            idx_c = c1*7 + c2
                            V4[idx_v][idx_c] = v1 * v2

    for k1 in range(2):
        for i1 in range(2):
            for c1 in range(7):
                w1 = W2[k1*2+i1][c1]
                if w1 == 0: continue
                for k2 in range(2):
                    for i2 in range(2):
                        for c2 in range(7):
                            w2 = W2[k2*2+i2][c2]
                            if w2 == 0: continue
                            idx_w = (k1*2 + k2)*4 + (i1*2 + i2)
                            idx_c = c1*7 + c2
                            W4[idx_w][idx_c] = w1 * w2

    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j][c] * V4[j*4+k][c] * W4[k*4+i][c]
                if s != 1:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    # Extend to 4x4x9
    U = [row[:] for row in U4]
    V = [[0]*49 for _ in range(36)]
    W = [[0]*49 for _ in range(36)]

    for j in range(4):
        for k in range(9):
            for c in range(49):
                V[j*9+k][c] = V4[j*4 + (k % 4)][c]

    for i in range(4):
        for k in range(9):
            for c in range(49):
                W[k*4+i][c] = W4[(k % 4)*4 + i][c]

    # Final Verification
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(49):
                    s += U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c]
                if s != 1:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U_f = [[0] * rank for _ in range(n * m)]
                    V_f = [[0] * rank for _ in range(m * p)]
                    W_f = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_f[ii*m+jj][col] = 1
                                V_f[jj*p+kk][col] = 1
                                W_f[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
