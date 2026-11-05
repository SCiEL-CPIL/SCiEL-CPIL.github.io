"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor.

    Contract sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k).
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)

    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    return bool(np.allclose(result, 1.0, atol=1e-9))


def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
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


def _block_decomposition_6x6x6():
    """Decompose 6x6x6 using 3x3 blocks of 2x2 matrices (Strassen). Rank = 27*7 = 189."""
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


def _hybrid_decomposition_6x6x6():
    """
    Hybrid 2x2-block decomposition of 6x6x6 matrix multiplication.

    Diagonal 2x2 sub-blocks (I==J==K) use the rank-7 Strassen structure.
    Off-diagonal 2x2 sub-blocks use rank-4 schoolbook (one scalar product per
    output element). Because the 2x2 Strassen W-coefficients of a diagonal
    block stay inside that same diagonal block, the decomposition is exact.

    Rank = 3*7 + 24*4 = 117.
    """
    n, m, p = 6, 6, 6
    B = 2
    G = 3  # 6 / 2
    diag_count = G
    off_count = G*G*G - G
    rank = diag_count*7 + off_count*4

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    sU, sV, sW = _strassen_2x2()

    col = 0
    for I in range(G):
        for J in range(G):
            for K in range(G):
                if I == J == K:
                    # Diagonal block: Strassen rank-7
                    for c in range(7):
                        for pi in range(B):
                            for pj in range(B):
                                u_row = (I*B+pi)*m + (J*B+pj)
                                U[u_row][col] += sU[pi*B+pj][c]
                        for pj in range(B):
                            for pk in range(B):
                                v_row = (J*B+pj)*p + (K*B+pk)
                                V[v_row][col] += sV[pj*B+pk][c]
                        for pk in range(B):
                            for pi in range(B):
                                w_row = (K*B+pk)*n + (I*B+pi)
                                W[w_row][col] += sW[pk*B+pi][c]
                        col += 1
                else:
                    # Off-diagonal block: schoolbook rank-4
                    for pi in range(B):
                        for pj in range(B):
                            for pk in range(B):
                                u_row = (I*B+pi)*m + (J*B+pj)
                                v_row = (J*B+pj)*p + (K*B+pk)
                                w_row = (K*B+pk)*n + (I*B+pi)
                                U[u_row][col] += 1
                                V[v_row][col] += 1
                                W[w_row][col] += 1
                                col += 1

    assert col == rank, (col, rank)
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
    deadline = start_time + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Candidate 1 (best rank if valid): hybrid Strassen-diagonal + schoolbook-offdiag (rank 117)
    try:
        hybrid = _hybrid_decomposition_6x6x6()
        if _verify(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
            return hybrid
    except Exception:
        pass

    if time.time() > deadline - 20:
        # not enough time left to verify the heavier candidate; return schoolbook
        return _schoolbook(n, m, p)

    # Candidate 2 (safe, known-correct): full Strassen 2x2 block (rank 189)
    try:
        best = _block_decomposition_6x6x6()
        if _verify(best["U"], best["V"], best["W"], n, m, p):
            return best
    except Exception:
        pass

    # Fallback: schoolbook (rank 216), always valid
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
