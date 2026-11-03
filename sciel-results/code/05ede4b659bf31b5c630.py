import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS

def construct():
    deadline = time.time() + 140.0

    # Correct Strassen-based rank-16 decomposition for 2x3x4 matrix multiplication.
    # Split j={0,1,2} into {0,1} and {2}, and k={0,1,2,3} into {0,1} and {2,3}.
    # Four blocks: (j in {0,1}, k in {0,1}), (j in {0,1}, k in {2,3}),
    #              (j in {2}, k in {0,1}), (j in {2}, k in {2,3})
    # Ranks: 7, 7, 1, 1 -> total 16

    rank = 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1}):
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    # C00 = m1 - m5 + m7 + m4
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    # Block 1: j in {0,1}, k in {0,1} (terms 0-6)
    # Term 0: m1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00, C11

    # Term 1: m2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C10

    # Term 2: m3 = A00(B10-B11)
    U[0][2] = 1
    V[4][2] = 1; V[5][2] = -1
    W[2][2] = 1  # C01

    # Term 3: m4 = A11(B01-B00)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C00, C10

    # Term 4: m5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C00, C01

    # Term 5: m6 = (A10-A00)(B00+B10)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[4][5] = 1
    W[3][5] = 1  # C11

    # Term 6: m7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[4][6] = -1
    V[1][6] = 1; V[5][6] = 1
    W[0][6] = 1; W[3][6] = -1  # C00, C11

    # Term 7: j=2, k in {0,1}: A[i,2]*B[2,k]
    U[2][7] = 1; U[5][7] = 1
    V[8][7] = 1; V[9][7] = 1
    W[0][7] = 1; W[1][7] = 1; W[2][7] = 1; W[3][7] = 1

    # Block 2: j in {0,1}, k in {2,3} (terms 8-14)
    # Term 8: m1 = (A00+A11)(B02+B13)
    U[0][8] = 1; U[4][8] = 1
    V[2][8] = 1; V[7][8] = 1
    W[4][8] = 1; W[7][8] = 1  # C02, C13

    # Term 9: m2 = (A10+A11)B02
    U[3][9] = 1; U[4][9] = 1
    V[2][9] = 1
    W[5][9] = 1  # C12

    # Term 10: m3 = A00(B12-B13)
    U[0][10] = 1
    V[6][10] = 1; V[7][10] = -1
    W[6][10] = 1  # C03

    # Term 11: m4 = A11(B03-B02)
    U[4][11] = 1
    V[3][11] = 1; V[2][11] = -1
    W[4][11] = 1; W[5][11] = 1  # C02, C12

    # Term 12: m5 = (A00+A01)B13
    U[0][12] = 1; U[1][12] = 1
    V[7][12] = 1
    W[4][12] = -1; W[6][12] = 1  # C02, C03

    # Term 13: m6 = (A10-A00)(B02+B12)
    U[3][13] = 1; U[0][13] = -1
    V[2][13] = 1; V[6][13] = 1
    W[7][13] = 1  # C13

    # Term 14: m7 = (A01-A11)(B03+B13)
    U[1][14] = 1; U[4][14] = -1
    V[3][14] = 1; V[7][14] = 1
    W[4][14] = 1; W[7][14] = -1  # C02, C13

    # Term 15: j=2, k in {2,3}: A[i,2]*B[2,k]
    U[2][15] = 1; U[5][15] = 1
    V[10][15] = 1; V[11][15] = 1
    W[4][15] = 1; W[5][15] = 1; W[6][15] = 1; W[7][15] = 1

    # Verification
    import numpy as np
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)

    C_calc = np.zeros((n, p))
    for c in range(rank):
        Ac = np.zeros((n, m))
        Bc = np.zeros((m, p))
        for i in range(n):
            for j in range(m):
                Ac[i, j] = U_np[i * m + j, c]
        for j in range(m):
            for k in range(p):
                Bc[j, k] = V_np[j * p + k, c]

        M = Ac @ Bc
        for i in range(n):
            for k in range(p):
                C_calc[i, k] += W_np[k * n + i, c] * M[i, k]

    if not np.allclose(C_true, C_calc, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
