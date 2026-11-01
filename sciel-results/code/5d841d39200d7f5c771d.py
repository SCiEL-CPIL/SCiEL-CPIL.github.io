"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x14 matrix.
Uses Strassen's algorithm on 2x2 blocks of the second matrix.
Total rank = 7 blocks * 7 Strassen terms = 49.
"""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-49 decomposition using Strassen's algorithm on 2x2 blocks.
    
    The 2x14 matrix B is viewed as 7 blocks of 2x2. Each block is multiplied
    by the 2x2 matrix A using Strassen's 7-term algorithm.
    Total rank = 7 * 7 = 49.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    t0 = time.time()
    t_limit = t0 + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nb = p // 2  # 7 blocks of 2x2
    R = nb * 7  # 49

    # Initialize matrices with zeros
    U = [[0] * R for _ in range(n * m)]  # 4 x 49
    V = [[0] * R for _ in range(m * p)]  # 28 x 49
    W = [[0] * R for _ in range(p * n)]  # 28 x 49

    # Strassen terms for 2x2 matrix multiplication
    # Each term is a tuple of 12 coefficients:
    # (a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
    # representing the contribution to the product C = A * B
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
            # A is 2x2, so indices are (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
            U[0][c] = a00
            U[1][c] = a01
            U[2][c] = a10
            U[3][c] = a11

            # Fill V: V[j*p+k, c]
            # B is 2x14. Block blk covers columns 2*blk and 2*blk+1.
            # j=0: rows 0 of A, so we look at B[0, 2*blk] and B[0, 2*blk+1]
            # j=1: rows 1 of A, so we look at B[1, 2*blk] and B[1, 2*blk+1]
            V[0 * p + 2 * blk][c] = b00
            V[0 * p + 2 * blk + 1][c] = b01
            V[1 * p + 2 * blk][c] = b10
            V[1 * p + 2 * blk + 1][c] = b11

            # Fill W: W[k*n+i, c]
            # C is 2x14. Block blk covers columns 2*blk and 2*blk+1.
            # k=0 (col 2*blk): C[0, 2*blk] and C[1, 2*blk]
            # k=1 (col 2*blk+1): C[0, 2*blk+1] and C[1, 2*blk+1]
            # Index in W is k*n + i
            # For col 2*blk (k_local=0):
            #   i=0 -> index 0*n + 0 = 0
            #   i=1 -> index 0*n + 1 = 1
            # For col 2*blk+1 (k_local=1):
            #   i=0 -> index 1*n + 0 = 2
            #   i=1 -> index 1*n + 1 = 3
            # Global index for k is 2*blk + k_local
            # So W index is (2*blk + k_local)*n + i
            
            W[(2 * blk + 0) * n + 0][c] = c00
            W[(2 * blk + 0) * n + 1][c] = c10
            W[(2 * blk + 1) * n + 0][c] = c01
            W[(2 * blk + 1) * n + 1][c] = c11

    if time.time() > t_limit:
        # Return best found (which is the complete construction)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
