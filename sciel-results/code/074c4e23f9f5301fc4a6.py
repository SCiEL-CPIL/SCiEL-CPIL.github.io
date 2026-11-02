import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 6

    def verify_full(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # ---- Winograd rank-7 decomposition for 2x2*2x2 ----
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U22 = [[0.0] * 7 for _ in range(4)]
    V22 = [[0.0] * 7 for _ in range(4)]
    W22 = [[0.0] * 7 for _ in range(4)]
    # term 0: (a00+a11)*(b00+b11) -> c00, c11
    U22[0][0] = 1.0; U22[3][0] = 1.0
    V22[0][0] = 1.0; V22[3][0] = 1.0
    W22[0][0] = 1.0; W22[3][0] = 1.0
    # term 1: (a10+a11)*b00 -> c10
    U22[2][1] = 1.0; U22[3][1] = 1.0
    V22[0][1] = 1.0
    W22[2][1] = 1.0
    # term 2: a00*(b01-b11) -> c01
    U22[0][2] = 1.0
    V22[1][2] = 1.0; V22[3][2] = -1.0
    W22[1][2] = 1.0
    # term 3: a11*(b10-b00) -> c10
    U22[3][3] = 1.0
    V22[2][3] = 1.0; V22[0][3] = -1.0
    W22[2][3] = 1.0
    # term 4: (a00+a01)*b11 -> c00(-), c01
    U22[0][4] = 1.0; U22[1][4] = 1.0
    V22[3][4] = 1.0
    W22[0][4] = -1.0; W22[1][4] = 1.0
    # term 5: (a10-a00)*(b00+b01) -> c11
    U22[2][5] = 1.0; U22[0][5] = -1.0
    V22[0][5] = 1.0; V22[1][5] = 1.0
    W22[3][5] = 1.0
    # term 6: (a01-a11)*(b10+b11) -> c00, c11
    U22[1][6] = 1.0; U22[3][6] = -1.0
    V22[2][6] = 1.0; V22[3][6] = 1.0
    W22[0][6] = 1.0; W22[3][6] = 1.0

    # ---- Rank-5 decomposition for 2x2*2x3 ----
    # A: 2x2, B: 2x3, C: 2x3
    # Indexing: U[i*2+j], V[j*3+k], W[k*2+i]
    U23 = [[0.0] * 5 for _ in range(4)]
    V23 = [[0.0] * 5 for _ in range(6)]
    W23 = [[0.0] * 5 for _ in range(6)]
    # t0: (a00+a11)*(b00+b10) -> c00, c11
    U23[0][0] = 1.0; U23[3][0] = 1.0
    V23[0][0] = 1.0; V23[3][0] = 1.0
    W23[0][0] = 1.0; W23[3][0] = 1.0
    # t1: (a00-a11)*(b01+b11) -> c01, -c11
    U23[0][1] = 1.0; U23[3][1] = -1.0
    V23[1][1] = 1.0; V23[4][1] = 1.0
    W23[1][1] = 1.0; W23[4][1] = -1.0
    # t2: (a01+a10)*(b02+b12) -> c02, c12
    U23[1][2] = 1.0; U23[2][2] = 1.0
    V23[2][2] = 1.0; V23[5][2] = 1.0
    W23[2][2] = 1.0; W23[5][2] = 1.0
    # t3: a00*b00 -> c00
    U23[0][3] = 1.0
    V23[0][3] = 1.0
    W23[0][3] = 1.0
    # t4: a11*b10 -> c11
    U23[3][4] = 1.0
    V23[3][4] = 1.0
    W23[3][4] = 1.0

    # Verify 2x2*2x3
    def verify_2x2_2x3(U, V, W):
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = np.dot(Ua[i * 2 + j], Va[j * 3 + k] * Wa[k * 2 + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    ok23 = verify_2x2_2x3(U23, V23, W23)

    # ---- Kronecker product: 4x4*4x6 = (2x2*2x2) ⊗ (2x2*2x3) ----
    # Rank = 7 * 5 = 35
    if ok23 and time.time() < deadline - 20:
        rank = 35
        U = [[0.0] * rank for _ in range(16)]
        V = [[0.0] * rank for _ in range(24)]
        W = [[0.0] * rank for _ in range(24)]

        # Kronecker product of decompositions
        # For each pair (t22, t23), the combined term index is t22*5 + t23
        for t22 in range(7):
            for t23 in range(5):
                r = t22 * 5 + t23
                # U: (2x2)⊗(2x2) -> 4x4
                for i22 in range(2):
                    for j22 in range(2):
                        for i23 in range(2):
                            for j23 in range(2):
                                # Full indices: i = i22*2 + i23, j = j22*2 + j23
                                i = i22 * 2 + i23
                                j = j22 * 2 + j23
                                U[i * 4 + j][r] = U22[i22 * 2 + j22][t22] * U23[i23 * 2 + j23][t23]
                # V: (2x2)⊗(2x3) -> 4x6
                for j22 in range(2):
                    for k22 in range(2):
                        for j23 in range(2):
                            for k23 in range(3):
                                j = j22 * 2 + j23
                                k = k22 * 2 + k23
                                V[j * 6 + k][r] = V22[j22 * 2 + k22][t22] * V23[j23 * 3 + k23][t23]
                # W: (2x2)⊗(2x3) -> 6x4 (k*n + i)
                for k22 in range(2):
                    for i22 in range(2):
                        for k23 in range(3):
                            for i23 in range(2):
                                k = k22 * 2 + k23
                                i = i22 * 2 + i23
                                W[k * 4 + i][r] = W22[k22 * 2 + i22][t22] * W23[k23 * 2 + i23][t23]

        if verify_full(U, V, W, 4, 4, 6):
            return {"U": U, "V": V, "W": W}

    # ---- Fallback: schoolbook rank 96 ----
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
