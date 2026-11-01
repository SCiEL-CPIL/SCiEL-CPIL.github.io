"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    # Rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    # Uses Strassen for the first 6 columns of B (3 blocks of 2), standard for column 6.
    # W convention: W[k*n + i, r] = coefficient for C[i,k] from product r.
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 4 x 25
    V = [[0] * rank for _ in range(m * p)]  # 14 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25

    # Standard term for column 6 (k=6): a[i,j] * b[j,6] -> c[i,6]
    # U[0,24]=1 (a00), U[1,24]=1 (a01), U[2,24]=1 (a10), U[3,24]=1 (a11)
    U[0][24] = 1  # a00
    U[1][24] = 1  # a01
    U[2][24] = 1  # a10
    U[3][24] = 1  # a11
    # V[7*0+6, 24]=1 (b06), V[7*1+6, 24]=1 (b16)
    V[6][24] = 1   # b[0,6]
    V[13][24] = 1  # b[1,6]
    # W[6*2+0, 24]=1 (c[0,6] gets a00*b06), W[6*2+1, 24]=1 (c[1,6] gets a10*b16)
    # Wait, c[0,6] = a[0,0]*b[0,6] + a[0,1]*b[1,6]
    # c[1,6] = a[1,0]*b[0,6] + a[1,1]*b[1,6]
    # So W[6*2+0, 24] should be coefficient for c[0,6]. The product r=24 gives a[i,j]*b[j,6].
    # For c[0,6]: a[0,0]*b[0,6] -> U[0]*V[6], a[0,1]*b[1,6] -> U[1]*V[13]
    # For c[1,6]: a[1,0]*b[0,6] -> U[2]*V[6], a[1,1]*b[1,6] -> U[3]*V[13]
    # W[2k+i, r] is the coefficient of product r in c[i,k].
    # c[0,6] = 1 * (U[0]*V[6] + U[1]*V[13])? No, W[2*6+0, 24] = 1 means c[0,6] += 1 * prod[24]
    # But prod[24] = U[0]*V[6] + U[1]*V[13] + U[2]*V[6] + U[3]*V[13]
    #             = a00*b06 + a01*b16 + a10*b06 + a11*b16
    # That's not right. We need separate products for each term.
    # Actually, for the standard 2x2*2x1 multiplication, we need 4 products:
    # p1 = a00*b06, p2 = a01*b16, p3 = a10*b06, p4 = a11*b16
    # Then c[0,6] = p1+p2, c[1,6] = p3+p4.
    # But we only have 1 column (rank 1) for this. That's not enough.
    # Wait, the schoolbook for 2x2*2x1 uses rank 4. So for column 6 alone, we need 4 columns.
    # But we're using rank 25 = 3*7 + 4. 3 Strassen blocks (6 cols) use 21 columns, remaining 4 for col 6.
    # Let me reconsider. Rank 25 = 21 (3 Strassen) + 4 (standard for 1 col).
    # But 21+4 = 25. Yes.

    # Let me redo. For column 6, use 4 standard terms (indices 21,22,23,24).
    # p21 = a00*b06: U[0,21]=1, V[6,21]=1, W[12,21]=1 (c[0,6]), 
    # p22 = a01*b16: U[1,22]=1, V[13,22]=1, W[12,22]=1 (c[0,6]),
    # p23 = a10*b06: U[2,23]=1, V[6,23]=1, W[13,23]=1 (c[1,6]),
    # p24 = a11*b16: U[3,24]=1, V[13,24]=1, W[13,24]=1 (c[1,6])

    # Column 6 standard terms (indices 21-24)
    U[0][21] = 1  # a00
    V[6][21] = 1  # b[0,6]
    W[12][21] = 1  # c[0,6]

    U[1][22] = 1  # a01
    V[13][22] = 1  # b[1,6]
    W[12][22] = 1  # c[0,6]

    U[2][23] = 1  # a10
    V[6][23] = 1  # b[0,6]
    W[13][23] = 1  # c[1,6]

    U[3][24] = 1  # a11
    V[13][24] = 1  # b[1,6]
    W[13][24] = 1  # c[1,6]

    # Strassen blocks for columns 0-5 (3 blocks)
    # Block b handles columns 2b, 2b+1
    # Strassen 7 products per block
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    #
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6

    for b in range(3):
        base = b * 7
        k0 = 2 * b
        k1 = 2 * b + 1

        # P1: (a00+a11)(b[0,k0]+b[1,k1])
        U[0][base + 0] = 1  # a00
        U[3][base + 0] = 1  # a11
        V[7 * 0 + k0][base + 0] = 1  # b[0,k0]
        V[7 * 1 + k1][base + 0] = 1  # b[1,k1]
        # c00 += P1, c11 += P1
        W[2 * k0 + 0][base + 0] = 1  # c[0,k0]
        W[2 * k1 + 1][base + 0] = 1  # c[1,k1]

        # P2: (a10+a11)b[0,k0]
        U[2][base + 1] = 1  # a10
        U[3][base + 1] = 1  # a11
        V[7 * 0 + k0][base + 1] = 1  # b[0,k0]
        # c10 += P2, c11 -= P2
        W[2 * k0 + 1][base + 1] = 1  # c[1,k0]
        W[2 * k1 + 1][base + 1] = -1  # c[1,k1]

        # P3: a00(b[0,k1]-b[1,k1])
        U[0][base + 2] = 1  # a00
        V[7 * 0 + k1][base + 2] = 1  # b[0,k1]
        V[7 * 1 + k1][base + 2] = -1  # b[1,k1]
        # c01 += P3, c11 += P3
        W[2 * k1 + 0][base + 2] = 1  # c[0,k1]
        W[2 * k1 + 1][base + 2] = 1  # c[1,k1]

        # P4: a11(b[1,k0]-b[0,k0])
        U[3][base + 3] = 1  # a11
        V[7 * 1 + k0][base + 3] = 1  # b[1,k0]
        V[7 * 0 + k0][base + 3] = -1  # b[0,k0]
        # c00 += P4, c10 += P4
        W[2 * k0 + 0][base + 3] = 1  # c[0,k0]
        W[2 * k0 + 1][base + 3] = 1  # c[1,k0]

        # P5: (a00+a01)b[1,k1]
        U[0][base + 4] = 1  # a00
        U[1][base + 4] = 1  # a01
        V[7 * 1 + k1][base + 4] = 1  # b[1,k1]
        # c00 -= P5, c01 += P5
        W[2 * k0 + 0][base + 4] = -1  # c[0,k0]
        W[2 * k1 + 0][base + 4] = 1  # c[0,k1]

        # P6: (a10-a00)(b[0,k0]+b[0,k1])
        U[2][base + 5] = 1  # a10
        U[0][base + 5] = -1  # a00
        V[7 * 0 + k0][base + 5] = 1  # b[0,k0]
        V[7 * 0 + k1][base + 5] = 1  # b[0,k1]
        # c11 += P6
        W[2 * k1 + 1][base + 5] = 1  # c[1,k1]

        # P7: (a01-a11)(b[1,k0]+b[1,k1])
        U[1][base + 6] = 1  # a01
        U[3][base + 6] = -1  # a11
        V[7 * 1 + k0][base + 6] = 1  # b[1,k0]
        V[7 * 1 + k1][base + 6] = 1  # b[1,k1]
        # c00 += P7
        W[2 * k0 + 0][base + 6] = 1  # c[0,k0]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
