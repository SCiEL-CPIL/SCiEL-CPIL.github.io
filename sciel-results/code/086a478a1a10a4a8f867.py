"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition using Strassen's algorithm on 2x2 blocks of the second matrix.
    
    The 2x14 matrix B is viewed as 7 blocks of 2x2. Each block is multiplied
    by the 2x2 matrix A using Strassen's 7-term algorithm.
    Total rank = 7 blocks * 7 Strassen terms = 49.
    
    The reconstruction is: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    This must equal sum_j A[i,j] * B[j,k].
    """
    t0 = time.time()
    t_limit = t0 + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nb = p // 2  # 7 blocks of 2x2
    R = nb * 7  # 49

    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # Strassen terms as full coefficient vectors
    # Each term: (a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
    # where c00=C[0,0], c01=C[0,1], c10=C[1,0], c11=C[1,1]
    terms = [
        # M1 = (A00+A11)(B00+B11)
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # M2 = (A10+A11)B00
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        # M3 = A00(B01-B11)
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        # M4 = A11(B10-B00)
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        # M5 = (A00+A01)B11
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # M6 = (A10-A00)(B00+B01)
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        # M7 = (A01-A11)(B10+B11)
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]

    for blk in range(nb):
        for s in range(7):
            c = 7 * blk + s
            a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11 = terms[s]

            # Fill U: U[i*m+j, c]
            U[0][c] = a00
            U[1][c] = a01
            U[2][c] = a10
            U[3][c] = a11

            # Fill V: V[j*p+k, c]
            V[0 * p + 2 * blk][c] = b00
            V[0 * p + 2 * blk + 1][c] = b01
            V[1 * p + 2 * blk][c] = b10
            V[1 * p + 2 * blk + 1][c] = b11

            # Fill W: W[k*n+i, c]
            W[2 * blk * n + 0][c] = c00
            W[(2 * blk + 1) * n + 0][c] = c01
            W[2 * blk * n + 1][c] = c10
            W[(2 * blk + 1) * n + 1][c] = c11

    if time.time() > t_limit:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
