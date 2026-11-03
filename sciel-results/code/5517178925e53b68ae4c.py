import numpy as np
import time

def _verify(n, m, p, U, V, W, tol=1e-6):
    """Verify the decomposition."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    np.random.seed(42)
    for _ in range(150):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        if abs(val) > tol:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = 6, 6, 6
    # Strassen 2x2 block decomposition with schoolbook 3x3: rank = 7 * 27 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 blocks
    # beta[s][I][J]: coeff of A[I,J] in the s-th product
    # gamma[s][K][L]: coeff of B[K,L] in the s-th product
    # alpha[s][I][K]: coeff of P[s] in C[I,K]

    beta = [[0]*2 for _ in range(2)]
    gamma = [[0]*2 for _ in range(2)]
    alpha = [[0]*2 for _ in range(2)]

    # s=0: (A00+A11)(B00+B11)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1

    # s=1: (A10+A11)B00
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1

    # s=2: A00(B10-B11)
    beta[2][0][0] = 1
    gamma[2][1][0] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1

    # s=3: A11(B01-B00)
    beta[3][1][1] = 1
    gamma[3][0][1] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1

    # s=4: (A00+A01)B11
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1

    # s=5: (A10-A00)(B00+B10)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][1][0] = 1
    alpha[5][1][1] = 1

    # s=6: (A01-A11)(B01+B11)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][0][1] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = -1; alpha[6][0][1] = 1

    # Build U, V, W using the block structure
    # Blocks are 3x3, so global index = block_idx * 3 + local_idx
    for s in range(7):
        for a in range(3):  # local row in block
            for b in range(3):  # local col in block
                for c in range(3):  # local col2 in block
                    r = s * 27 + a * 9 + b * 3 + c

                    # U: A components
                    for I in range(2):
                        for J in range(2):
                            if beta[s][I][J] != 0:
                                i_glob = I * 3 + a
                                j_glob = J * 3 + b
                                U[i_glob * m + j_glob][r] += beta[s][I][J]

                    # V: B components
                    for K in range(2):
                        for L in range(2):
                            if gamma[s][K][L] != 0:
                                j_glob = K * 3 + b
                                k_glob = L * 3 + c
                                V[j_glob * p + k_glob][r] += gamma[s][K][L]

                    # W: C components (trace convention W[k*n+i])
                    for I in range(2):
                        for K in range(2):
                            if alpha[s][I][K] != 0:
                                k_glob = K * 3 + c
                                i_glob = I * 3 + a
                                W[k_glob * n + i_glob][r] += alpha[s][I][K]

    # Check deadline
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    # Verify
    if _verify(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook 216
    U_sb = [[0] * 216 for _ in range(36)]
    V_sb = [[0] * 216 for _ in range(36)]
    W_sb = [[0] * 216 for _ in range(36)]
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U_sb[i*6+j][col] = 1
                V_sb[j*6+k][col] = 1
                W_sb[k*6+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
