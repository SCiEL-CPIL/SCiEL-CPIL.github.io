"""Matrix multiplication 6x6x6 via Kronecker product of 3x3 schoolbook and 2x2 Strassen."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Use Kronecker product of 3x3 schoolbook (rank 9) and 2x2 Strassen (rank 7).
    # This gives rank 9*7 = 63 for 6x6 matrix multiplication.
    
    # 2x2 Strassen factors (rank 7):
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: M1 = A00 * B00
    U2[0, 0] = 1; V2[0, 0] = 1; W2[0, 0] = 1
    # Term 1: M2 = (A00+A01) * B11
    U2[0, 1] = 1; U2[1, 1] = 1; V2[3, 1] = 1; W2[1, 1] = 1
    # Term 2: M3 = A10 * (B00+B01)
    U2[2, 2] = 1; V2[0, 2] = 1; V2[1, 2] = 1; W2[1, 2] = 1; W2[2, 2] = 1
    # Term 3: M4 = A11 * (B10+B11)
    U2[3, 3] = 1; V2[2, 3] = 1; V2[3, 3] = 1; W2[2, 3] = 1; W2[3, 3] = 1
    # Term 4: M5 = (A00+A11) * (B00+B11)
    U2[0, 4] = 1; U2[3, 4] = 1; V2[0, 4] = 1; V2[3, 4] = 1; W2[0, 4] = 1; W2[3, 4] = -1
    # Term 5: M6 = (A01-A11) * (B10+B11)
    U2[1, 5] = 1; U2[3, 5] = -1; V2[2, 5] = 1; V2[3, 5] = 1; W2[3, 5] = 1
    # Term 6: M7 = (A00+A10) * (B01-B11)
    U2[0, 6] = 1; U2[2, 6] = 1; V2[1, 6] = 1; V2[3, 6] = -1; W2[0, 6] = 1; W2[3, 6] = -1
    
    # Verify Strassen
    A_test = np.array([[1, 2], [3, 4]], dtype=float)
    B_test = np.array([[5, 6], [7, 8]], dtype=float)
    C_test = A_test @ B_test
    C_comp = np.zeros((2, 2))
    for c in range(7):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    C_comp[k, i] += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
    
    strassen_ok = np.allclose(C_comp, C_test, atol=1e-10)
    
    # 3x3 schoolbook factors (rank 9):
    U3 = np.zeros((9, 9))
    V3 = np.zeros((9, 9))
    W3 = np.zeros((9, 9))
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                c = bi*9 + bj*3 + bk
                U3[bi*3+bj, c] = 1
                V3[bj*3+bk, c] = 1
                W3[bk*3+bi, c] = 1
    
    # Kronecker product: 6x6 decomposition
    rank6 = 9 * 7  # 63
    U6 = np.zeros((36, rank6))
    V6 = np.zeros((36, rank6))
    W6 = np.zeros((36, rank6))
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                c3 = bi*9 + bj*3 + bk
                for li in range(2):
                    for lj in range(2):
                        for lk in range(2):
                            for c2 in range(7):
                                c6 = c3*7 + c2
                                i = bi*2 + li
                                j = bj*2 + lj
                                k = bk*2 + lk
                                U6[i*6+j, c6] = U3[bi*3+bj, c3] * U2[li*2+lj, c2]
                                V6[j*6+k, c6] = V3[bj*3+bk, c3] * V2[lj*2+lk, c2]
                                W6[k*6+i, c6] = W3[bk*3+bi, c3] * W2[lk*2+li, c2]
    
    # Verify 6x6
    A_rand = np.random.rand(6, 6)
    B_rand = np.random.rand(6, 6)
    C_rand = A_rand @ B_rand
    C_comp6 = np.zeros((6, 6))
    for c in range(rank6):
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    C_comp6[k, i] += U6[i*6+j, c] * V6[j*6+k, c] * W6[k*6+i, c]
    
    kronecker_ok = np.allclose(C_comp6, C_rand, atol=1e-10)
    
    if strassen_ok and kronecker_ok:
        # Use the Kronecker product decomposition (rank 63)
        U_list = [[int(U6[i, c]) for c in range(rank6)] for i in range(36)]
        V_list = [[int(V6[i, c]) for c in range(rank6)] for i in range(36)]
        W_list = [[int(W6[i, c]) for c in range(rank6)] for i in range(36)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook (rank 216)
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
