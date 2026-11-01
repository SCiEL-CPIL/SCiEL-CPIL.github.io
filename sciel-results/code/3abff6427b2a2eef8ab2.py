"""Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""

def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.

    Tensor convention:
        U[i*m + j, c]  for A[i, j]
        V[j*p + k, c]  for B[j, k]
        W[k*n + i, c]  for C[i, k]

    with n = m = p = 2.

    The rank is 7, which is the proven theoretical minimum for 2x2 matrix multiplication
    (Strassen's lower bound). This is an exact decomposition with integer entries.
    """
    rank = 7

    # Strassen's seven products:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    #
    # m0 = (a00 + a11)(b00 + b11)
    # m1 = (a10 + a11)b00
    # m2 = a00(b01 - b11)
    # m3 = a11(b10 - b00)
    # m4 = (a00 + a01)b11
    # m5 = (a10 - a00)(b00 + b01)
    # m6 = (a01 - a11)(b10 + b11)
    #
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    #
    # Index mapping:
    # U row: i*m + j  => a00:0, a01:1, a10:2, a11:3
    # V row: j*p + k  => b00:0, b01:1, b10:2, b11:3
    # W row: k*n + i  => c00:0, c10:1, c01:2, c11:3

    U = [
        [1,  0,  1,  0,  1, -1,  0],   # a00
        [0,  0,  0,  0,  1,  0,  1],   # a01
        [0,  1,  0,  0,  0,  1,  0],   # a10
        [1,  1,  0,  1,  0,  0, -1],   # a11
    ]
    V = [
        [1,  1,  0, -1,  0,  1,  0],   # b00
        [0,  0,  1,  0,  0,  1,  0],   # b01
        [0,  0,  0,  1,  0,  0,  1],   # b10
        [1,  0, -1,  0,  1,  0,  1],   # b11
    ]
    W = [
        [1,  0,  0,  1, -1,  0,  1],   # c00
        [0,  1,  0,  1,  0,  0,  0],   # c10
        [0,  0,  1,  0,  1,  0,  0],   # c01
        [1, -1,  1,  0,  0,  1,  0],   # c11
    ]

    return {"U": U, "V": V, "W": W}
