"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Exploit 6 = 2*3 structure: C = (A2 x A3)(B2 x B3).
    # 2x2 block (i2,k2) handled by rank-7 Strassen; 3x3 block (i3,k3) by rank-9 schoolbook.
    # Total rank = 7 * 9 = 63.
    #
    # Strassen 2x2 (indices 0,1):
    #   M0 = (A00+A11)(B00+B11)
    #   M1 = (A10+A11)B00
    #   M2 = A00(B10-B11)
    #   M3 = A11(B01-B00)
    #   M4 = (A00+A01)B11
    #   M5 = (A10-A00)(B00+B10)
    #   M6 = (A01-A11)(B01+B11)
    #   C00 = M0+M3-M4+M6 ; C01 = M2+M4
    #   C10 = M1+M5 ;       C11 = M0-M1+M2+M5
    strassen_U = [
        # (a00,a11,a10,a11,a00,0,0,0,0,0)
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    ]
    strassen_V = [
        # (b00,b11,b00,b00,b11,b00,b11,b10,b01,b11)
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    ]
    strassen_W = [
        # (c00,c01,c10,c11)
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
    ]

    rank = 7 * 9
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    col = 0
    for r in range(7):
        for t in range(9):
            # 3x3 schoolbook: t = i3*3 + k3, factor over j3
            i3 = t // 3
            k3 = t % 3
            for j3 in range(3):
                u2 = strassen_U[r]
                v2 = strassen_V[r]
                for i2 in range(2):
                    for j2 in range(2):
                        a = u2[i2 * 2 + j2]
                        if a:
                            U[i2 * 3 + i3][col] += a
                for j2 in range(2):
                    for k2 in range(2):
                        b = v2[j2 * 2 + k2]
                        if b:
                            V[j2 * 3 + j3][col] += b
                # W: product M_r * (A[i3,j3]*B[j3,k3]) -> C[i2,k2][i3,k3]
                for i2 in range(2):
                    for k2 in range(2):
                        c = strassen_W[r][i2 * 2 + k2]
                        if c:
                            W[k2 * 3 + k3][col] += c
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
