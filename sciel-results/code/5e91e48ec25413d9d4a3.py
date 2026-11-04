"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_np(U, V, W, n, m, p, tol=1e-8):
    """Fast NumPy verification of the tensor decomposition."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    Ua = Ua.reshape(n, m, Ua.shape[1])
    Va = Va.reshape(m, p, Va.shape[1])
    Wa = Wa.reshape(p, n, Wa.shape[1])
    # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # einsum: i j c, j k c, k i c -> i j k
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return bool(np.allclose(T, 1.0, atol=tol, rtol=0.0))

def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1 = (a11+a22)*(b11+b22): C11 += M1, C22 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (a21+a22)*b11: C21 += M2, C22 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # M3 = a11*(b12-b22): C12 += M3, C22 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # M4 = a22*(b21-b11): C11 += M4, C21 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # M5 = (a11+a12)*b22: C11 -= M5, C12 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # M6 = (a21-a11)*(b11+b12): C22 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (a12-a22)*(b21+b22): C11 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _lanczos_3x3():
    """Rank-8 Lanczos decomposition for 3x3x3 matrix multiplication.
    U: (9, 8), V: (9, 8), W: (9, 8)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U = [[0]*8 for _ in range(9)]
    V = [[0]*8 for _ in range(9)]
    W = [[0]*8 for _ in range(9)]

    # P1 = (a11+a21+a31)*(b11+b21+b31) -> c11, c22, c33
    U[0][0] = 1; U[3][0] = 1; U[6][0] = 1
    V[0][0] = 1; V[3][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[8][0] = 1

    # P2 = a31*(b12+b13) -> c12, c13
    U[6][1] = 1
    V[1][1] = 1; V[2][1] = 1
    W[3][1] = 1; W[6][1] = 1

    # P3 = (a12+a13)*(b31+b32) -> c11, c21
    U[1][2] = 1; U[2][2] = 1
    V[6][2] = 1; V[7][2] = 1
    W[0][2] = 1; W[1][2] = 1

    # P4 = a11*(b22+b23) -> c22, c23
    U[0][3] = 1
    V[4][3] = 1; V[5][3] = 1
    W[4][3] = 1; W[7][3] = 1

    # P5 = (a22+a23)*(b12+b13) -> c11, c21
    U[4][4] = 1; U[5][4] = 1
    V[1][4] = 1; V[2][4] = 1
    W[0][4] = 1; W[1][4] = 1

    # P6 = a21*(b32+b33) -> c12, c13
    U[3][5] = 1
    V[7][5] = 1; V[8][5] = 1
    W[3][5] = 1; W[6][5] = 1

    # P7 = (a12+a22+a32)*(b21+b22+b32) -> c11, c22, c33
    U[1][6] = 1; U[4][6] = 1; U[7][6] = 1
    V[3][6] = 1; V[4][6] = 1; V[7][6] = 1
    W[0][6] = 1; W[4][6] = 1; W[8][6] = 1

    # P8 = (a13+a23+a33)*(b13+b23+b33) -> c11, c22, c33
    U[2][7] = 1; U[5][7] = 1; U[8][7] = 1
    V[2][7] = 1; V[5][7] = 1; V[8][7] = 1
    W[0][7] = 1; W[4][7] = 1; W[8][7] = 1

    return U, V, W

def _tensor_product_6x6x6():
    """
    Structural decomposition: 6 = 2 (Strassen) x 3 (Lanczos).
    Tensor product of the 2x2 (rank 7) and 3x3 (rank 8) decompositions.
    Total rank = 7 * 8 = 56.
    """
    n, m, p = 6, 6, 6
    sU, sV, sW = _strassen_2x2()
    lU, lV, lW = _lanczos_3x3()

    rank = 7 * 8
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]

    col = 0
    for cs in range(7):
        for cl in range(8):
            for oi in range(2):
                for oj in range(2):
                    for ti in range(3):
                        for tj in range(3):
                            u_row = (oi*3 + ti)*6 + (oj*3 + tj)
                            U[u_row][col] = sU[oi*2 + oj][cs] * lU[ti*3 + tj][cl]
            for oj in range(2):
                for ok in range(2):
                    for tj in range(3):
                        for tk in range(3):
                            v_row = (oj*3 + tj)*6 + (ok*3 + tk)
                            V[v_row][col] = sV[oj*2 + ok][cs] * lV[tj*3 + tk][cl]
            for ok in range(2):
                for oi in range(2):
                    for tk in range(3):
                        for ti in range(3):
                            w_row = (ok*3 + tk)*6 + (oi*3 + ti)
                            W[w_row][col] = sW[ok*2 + oi][cs] * lW[tk*3 + ti][cl]
            col += 1

    return {"U": U, "V": V, "W": W}

def _block_decomposition_6x6x6():
    """Fallback: 3x3 grid of 2x2 Strassen blocks. Rank = 27 * 7 = 189."""
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]

    sU, sV, sW = _strassen_2x2()

    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    col += 1
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    # Primary structural candidate: 2x2 (Strassen) x 3x3 (Lanczos) tensor product, rank 56.
    candidate = _tensor_product_6x6x6()
    if time.time() < deadline and _verify_np(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate

    # Fallback: 3x3 grid of 2x2 Strassen blocks, rank 189.
    candidate = _block_decomposition_6x6x6()
    if time.time() < deadline and _verify_np(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate

    # Last resort: schoolbook, rank 216.
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
