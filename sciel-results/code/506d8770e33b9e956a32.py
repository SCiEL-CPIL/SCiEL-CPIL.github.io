"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

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

def _schoolbook_3x3():
    """Rank-27 schoolbook decomposition for 3x3x3 matrix multiplication.
    U: (9, 27), V: (9, 27), W: (9, 27)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def _block_decomposition_6x6x6():
    """3x3 grid of 2x2 Strassen blocks with 3x3 schoolbook inner. Rank = 27 * 7 = 189."""
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]

    sU, sV, sW = _strassen_2x2()
    bU, bV, bW = _schoolbook_3x3()

    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    # Outer Strassen index c
                    for li in range(2):
                        for lj in range(2):
                            if sU[li*2+lj][c] != 0:
                                # Inner 3x3 block at (bi, bj) -> contributes to U[(bi*2+li)*6 + (bj*2+lj), col]
                                # Wait, the block is 2x2, so the row index within the 6x6 is (bi*2+li)*6 + (bj*2+lj)
                                # This is correct for 3 blocks of size 2.
                                # The inner 3x3 is actually a block of size 2? No, 6 = 3*2.
                                # So the "inner" dimension is 2, not 3.
                                # Ah, the current program uses 3x3 grid of 2x2 blocks.
                                # The inner block is 2x2.
                                # So I should use _strassen_2x2 for the inner block too? No, that would be rank 7*7*7 = 343.
                                # Or schoolbook 2x2 (rank 4) -> 7*4*7 = 196.
                                # Or Strassen 2x2 (rank 7) -> 7*7*7 = 343.
                                # The current program uses _strassen_2x2 for the blocks.
                                # So the inner block is 2x2.
                                # So I should iterate over the 2x2 inner block.
                                pass
    # This function is complex to rewrite from scratch without the original logic.
    # I will reuse the logic from the current program which is known to work and score 1.1429.
    # The current program uses _strassen_2x2 for the 2x2 blocks.
    # Rank = 27 blocks * 7 (strassen) = 189.
    # Let's copy the working logic.
    
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

def _verify_np(U, V, W, n, m, p, tol=1e-8):
    """Fast NumPy verification of the tensor decomposition."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    Ua = Ua.reshape(n, m, Ua.shape[1])
    Va = Va.reshape(m, p, Va.shape[1])
    Wa = Wa.reshape(p, n, Wa.shape[1])
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return bool(np.allclose(T, 1.0, atol=tol, rtol=0.0))

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

    # Primary structural candidate: 3x3 grid of 2x2 Strassen blocks, rank 189.
    candidate = _block_decomposition_6x6x6()
    if time.time() < deadline and _verify_np(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate

    # Last resort: schoolbook, rank 216.
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
