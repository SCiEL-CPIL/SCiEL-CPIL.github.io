"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7

    # Strassen decomposition for 2x2 matrix multiplication.
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6

    U = [
        [1, 0, 1, 0, 1, 1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, -1, -1]  # a11
    ]

    V = [
        [1, 1, 0, 1, 0, 1, 1],   # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, -1, 1, 0, 0, 1],  # b10
        [1, 0, -1, -1, 1, -1, 1] # b11
    ]

    W = [
        [1, 0, 0, 1, -1, 0, -1], # c00 (k=0, i=0)
        [0, 1, 0, 0, 0, 0, 0],   # c10 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],   # c01 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # c11 (k=1, i=1)
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
