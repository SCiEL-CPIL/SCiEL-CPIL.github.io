"""4x4 matrix multiplication tensor decomposition using recursive Strassen."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    
    A 4x4 matrix is viewed as a 2x2 matrix of 2x2 blocks.
    We apply Strassen's algorithm at the block level (7 block multiplications),
    and each block multiplication is a 2x2 matrix multiplication, which we
    also decompose using Strassen (7 scalar multiplications each).
    
    Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS  # all 4
    
    # Strassen decomposition for 2x2 matrix multiplication.
    # For X, Y in M_2(R), Z = X * Y:
    # m1 = (X00 + X11) * (Y00 + Y11)
    # m2 = (X10 + X11) * Y00
    # m3 = X00 * (Y01 - Y11)
    # m4 = X11 * (Y10 - Y00)
    # m5 = (X00 + X01) * Y11
    # m6 = (X10 - X00) * (Y00 + Y01)
    # m7 = (X01 - X11) * (Y10 + Y11)
    #
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
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
    strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # m1: Z00 += 1, Z11 += 1
        [(1, 1, 0), (-1, 1, 1)],      # m2: Z10 += 1, Z11 += -1
        [(1, 0, 1), (1, 1, 1)],       # m3: Z01 += 1, Z11 += 1
        [(1, 0, 0), (1, 1, 0)],       # m4: Z00 += 1, Z10 += 1
        [(-1, 0, 0), (1, 0, 1)],      # m5: Z00 += -1, Z01 += 1
        [(1, 1, 1)],                   # m6: Z11 += 1
        [(1, 0, 0)],                   # m7: Z00 += 1
    ]
    
    rank = 7 * 7  # 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    col = 0
    for outer_s in range(7):  # Outer Strassen term
        for inner_s in range(7):  # Inner Strassen term
            # Process U: combine outer and inner U factors
            for (outer_coeff, outer_br, outer_bc) in strassen_U[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_U[inner_s]:
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    U[gr * m + gc][col] += outer_coeff * inner_coeff
            
            # Process V: combine outer and inner V factors
            for (outer_coeff, outer_br, outer_bc) in strassen_V[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_V[inner_s]:
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    V[gr * p + gc][col] += outer_coeff * inner_coeff
            
            # Process W: combine outer and inner W factors
            for (outer_coeff, outer_br, outer_bc) in strassen_W[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_W[inner_s]:
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    # W[k*n+i] where k=col, i=row
                    W[gc * n + gr][col] += outer_coeff * inner_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
