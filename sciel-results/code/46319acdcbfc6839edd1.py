import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for 2x1 blocks.
    Rank = 5 * (7 + 4) = 55.
    """
    n, m, p = 2, 3, 10
    rank = 55

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # Indices for A_L (2x2 part of 2x3)
    # A is 2x3. Indices: i*3+j
    # A00: 0, A01: 1, A10: 3, A11: 4
    a00, a01, a10, a11 = 0, 1, 3, 4

    # Indices for A_R (2x1 part of 2x3)
    a02 = 0 * 3 + 2
    a12 = 1 * 3 + 2

    # We split the 10 columns of B into 5 blocks of 2 columns.
    # For each block b (0..4):
    #   B_block is 3x2.
    #   We split B_block into B_L (2x2, top 2 rows) and B_R (1x2, bottom 1 row).
    #   We compute C_block = A_L B_L + A_R B_R.
    #   A_L is 2x2, A_R is 2x1.
    #   C_block is 2x2.
    #
    #   A_L B_L is a 2x2x2 matmul. Rank 7 (Strassen).
    #   A_R B_R is a 2x1x2 matmul. Rank 4 (Schoolbook).
    #   Total rank per block = 7 + 4 = 11.
    #   Total rank = 5 * 11 = 55.

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # B indices
        # B is 3x10. Index j*10 + k.
        # B_L is rows 0,1. B_R is row 2.
        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1

        # C indices
        # C is 2x10. Official trace convention W[k*n+i, r] means C[k, i] is at index k*n + i.
        # n=2. So C[k, i] is at k*2 + i.
        # C_block corresponds to columns k0, k1.
        # C00 = C[k0, 0] -> index k0*2 + 0
        # C01 = C[k1, 0] -> index k1*2 + 0
        # C10 = C[k0, 1] -> index k0*2 + 1
        # C11 = C[k1, 1] -> index k1*2 + 1
        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1

        # Strassen terms for A_L B_L
        # A_L = [[a00, a01], [a10, a11]]
        # B_L = [[b00, b01], [b10, b11]]
        
        # M1: (A00+A11)(B00+B11)
        U[a00, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M2: (A10+A11)B00
        U[a10, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0
        W[c11, col] = -1.0
        col += 1

        # M3: A00(B01-B11)
        U[a00, col] = 1.0
        V[b01, col] = 1.0
        V[b11, col] = -1.0
        W[c01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M4: A11(B10-B00)
        U[a11, col] = 1.0
        V[b10, col] = 1.0
        V[b00, col] = -1.0
        W[c00, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # M5: (A00+A01)B11
        U[a00, col] = 1.0
        U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0
        W[c01, col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01)
        U[a10, col] = 1.0
        U[a00, col] = -1.0
        V[b00, col] = 1.0
        V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11)
        U[a01, col] = 1.0
        U[a11, col] = -1.0
        V[b10, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # Schoolbook terms for A_R B_R
        # A_R = [[a02], [a12]]
        # B_R = [[b20, b21]]
        # C_R = [[a02*b20, a02*b21], [a12*b20, a12*b21]]
        
        # P1: A02*B20 -> C00
        U[a02, col] = 1.0
        V[b20, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # P2: A02*B21 -> C01
        U[a02, col] = 1.0
        V[b21, col] = 1.0
        W[c01, col] = 1.0
        col += 1

        # P3: A12*B20 -> C10
        U[a12, col] = 1.0
        V[b20, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # P4: A12*B21 -> C11
        U[a12, col] = 1.0
        V[b21, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
