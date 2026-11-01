"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal (rank-7) Strassen tensor decomposition for 2x2 matmul.
    U shape: (n*m, rank) = (4, 7)
    V shape: (m*p, rank) = (4, 7)
    W shape: (p*n, rank) = (4, 7)
    Convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    rank = 7

    # Strassen's 7 intermediate products for:
    #   A = [[a, b], [c, d]] * B = [[e, f], [g, h]]
    #   m1 = (a+d)(e+h)
    #   m2 = (c+d)e
    #   m3 = a(f-h)
    #   m4 = d(g-e)
    #   m5 = (a+b)h
    #   m6 = (c-a)(e+f)
    #   m7 = (b-d)(g+h)
    #
    # Result:
    #   top-left    = m1 + m4 - m5 + m7
    #   top-right   = m3 + m5
    #   bottom-left = m2 + m4
    #   bottom-right= m1 - m2 + m3 + m6

    # U[i*m + j, r]: row index = i*m + j, r = 0..6
    U = [
        [1,  0,  1,  0,  1,  0,  0],  # a (i=0,j=0): m1, m3, m5
        [0,  0,  0,  0,  1, -1,  0],  # b (i=0,j=1): m5, m6
        [0,  1,  0,  0,  0,  1,  0],  # c (i=1,j=0): m2, m6
        [1,  1,  0,  1,  0,  0, -1]   # d (i=1,j=1): m1, m2, m4, m7
    ]

    # V[j*p + k, r]: row index = j*p + k, r = 0..6
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # e (j=0,k=0): m1, m2, m4, m6
        [0,  0,  0,  0,  0,  1,  1],  # f (j=0,k=1): m6, m7
        [0,  0,  1,  0,  0,  0,  1],  # g (j=1,k=0): m3, m7
        [1,  0, -1,  1,  1,  0,  0]   # h (j=1,k=1): m1, m3, m4, m5
    ]

    # W[k*n + i, r]: row index = k*n + i, r = 0..6
    # k=0,i=0 -> top-left; k=0,i=1 -> top-right; k=1,i=0 -> bottom-left; k=1,i=1 -> bottom-right
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # top-left
        [0,  0,  1,  0,  1,  0,  0],  # top-right
        [0,  1,  0,  1,  0,  0,  0],  # bottom-left
        [1, -1, 1,  0,  0,  1,  0]   # bottom-right
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
