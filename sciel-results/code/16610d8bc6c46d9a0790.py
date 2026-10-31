"""4x4 matrix multiplication tensor decomposition using Strassen on 2x2 blocks."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using the block structure with Strassen's algorithm.
    
    A 4x4 matrix is viewed as a 2x2 matrix of 2x2 blocks.
    C = A * B where:
      C00 = A00*B00 + A01*B10
      C01 = A00*B01 + A01*B11
      C10 = A10*B00 + A11*B10
      C11 = A10*B01 + A11*B11
    
    Each block product is a 2x2 matrix multiplication, decomposed using Strassen (rank 7).
    Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # all 4
    
    # Strassen U factors: for each of the 7 multiplications, list of (coeff, i, j) for X
    strassen_U = [
        [(1, 0, 0), (1, 1, 1)],       # m1: X00 + X11
        [(1, 1, 0), (1, 1, 1)],       # m2: X10 + X11
        [(1, 0, 0)],                   # m3: X00
        [(1, 1, 1)],                   # m4: X11
        [(1, 0, 0), (1, 0, 1)],       # m5: X00 + X01
        [(1, 1, 0), (-1, 0, 0)],      # m6: X10 - X00
        [(1, 0, 1), (-1, 1, 1)],      # m7: X01 - X11
    ]
    
    # Strassen V factors: for each of the 7 multiplications, list of (coeff, i, j) for Y
    strassen_V = [
        [(1, 0, 0), (1, 1, 1)],       # m1: Y00 + Y11
        [(1, 0, 0)],                   # m2: Y00
        [(1, 0, 1), (-1, 1, 1)],      # m3: Y01 - Y11
        [(1, 1, 0), (-1, 0, 0)],      # m4: Y10 - Y00
        [(1, 1, 1)],                   # m5: Y11
        [(1, 0, 0), (1, 0, 1)],       # m6: Y00 + Y01
        [(1, 1, 0), (1, 1, 1)],       # m7: Y10 + Y11
    ]
    
    # Strassen W factors: for each of the 7 multiplications, list of (coeff, i, j) for Z
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 1, 0), (-1, 1, 1)],      # m2
        [(1, 0, 1), (1, 1, 1)],       # m3
        [(1, 0, 0), (1, 1, 0)],       # m4
        [(-1, 0, 0), (1, 0, 1)],      # m5
        [(1, 1, 1)],                   # m6
        [(1, 0, 0)],                   # m7
    ]
    
    def block_offset(block_row, block_col, local_row, local_col):
        """Convert block coordinates to global coordinates."""
        return 2 * block_row + local_row, 2 * block_col + local_col
    
    rank = 8 * 7  # 56
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 16 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for a in range(2):  # block row of C
        for b in range(2):  # block col of C
            for k_block in range(2):  # which term in the sum
                # A block is (a, k_block), B block is (k_block, b)
                for s in range(7):  # Strassen term
                    # U factors
                    for (coeff, lr, lc) in strassen_U[s]:
                        gr, gc = block_offset(a, k_block, lr, lc)
                        U[gr * m + gc][col] += coeff
                    
                    # V factors
                    for (coeff, lr, lc) in strassen_V[s]:
                        gr, gc = block_offset(k_block, b, lr, lc)
                        V[gr * p + gc][col] += coeff
                    
                    # W factors
                    for (coeff, lr, lc) in strassen_W[s]:
                        gr, gc = block_offset(a, b, lr, lc)
                        W[gc * n + gr][col] += coeff  # W[k*n+i] where k=col, i=row
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
