"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B = [[ae+bg, af+bh], [ce+dg, cf+dh]]

    # U maps (a,b,c,d) -> U[i*m+j, r], i.e., U[row, col]
    # row = i*m + j, where i indexes rows of A, j indexes cols of A
    # V maps (e,f,g,h) -> V[j*p+k, r], i.e., V[row, col]
    # row = j*p + k, where j indexes rows of B, k indexes cols of B
    # W maps output to W[k*n+i, r], i.e., W[row, col]
    # row = k*n + i, where k indexes cols of C, i indexes rows of C

    # Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)

    # U: shape (4, 7) - rows for a, b, c, d
    U = [
        [1,  0,  1,  0,  1,  1,  0],  # a
        [0,  0,  0,  0,  1,  0,  1],  # b
        [0,  1,  0,  0,  0, -1,  0],  # c
        [1,  1,  0,  1,  0,  0, -1],  # d
    ]

    # V: shape (4, 7) - rows for e, f, g, h
    V = [
        [1,  1,  0,  0,  0,  1,  0],  # e
        [0,  0,  1, -1,  0,  1,  0],  # f
        [0,  0,  0,  1,  0,  0,  1],  # g
        [1,  0, -1,  0,  1,  0,  1],  # h
    ]

    # W: shape (4, 7) - rows indexed by k*n+i = k*2+i
    # Row 0: i=0, k=0 -> C[0][0] = ae+bg
    # Row 1: i=1, k=0 -> C[1][0] = ce+dg
    # Row 2: i=0, k=1 -> C[0][1] = af+bh
    # Row 3: i=1, k=1 -> C[1][1] = cf+dh

    # C[0][0] = M1 + M4 - M5 + M7
    # C[1][0] = M2 + M4
    # C[0][1] = M3 + M5
    # C[1][1] = M1 - M2 + M3 + M6

    W = [
        [1,  0,  0,  1, -1,  0,  1],  # C[0][0]: M1 + M4 - M5 + M7
        [0,  1,  0,  1,  0,  0,  0],  # C[1][0]: M2 + M4
        [0,  0,  1,  0,  1,  0,  0],  # C[0][1]: M3 + M5
        [1, -1, 1,  0,  0,  1,  0],  # C[1][1]: M1 - M2 + M3 + M6
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
