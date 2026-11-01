"""Tensor decomposition for 2x2 by 2x5 matrix multiplication."""

def construct():
    n, m, p = 2, 2, 5
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t_offset, k_start):
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: j*p + k
        k0 = k_start
        k1 = k_start + 1

        # M1: A00+A11, B00+B11
        U[0][t_offset+0] = 1
        U[3][t_offset+0] = 1
        V[0*p+k0][t_offset+0] = 1
        V[1*p+k1][t_offset+0] = 1
        # C00 += M1, C11 += M1
        W[k0*n+0][t_offset+0] = 1
        W[k1*n+1][t_offset+0] = 1

        # M2: A10+A11, B00
        U[2][t_offset+1] = 1
        U[3][t_offset+1] = 1
        V[0*p+k0][t_offset+1] = 1
        # C10 += M2, C11 -= M2
        W[k0*n+1][t_offset+1] = 1
        W[k1*n+1][t_offset+1] = -1

        # M3: A00, B01-B11
        U[0][t_offset+2] = 1
        V[0*p+k1][t_offset+2] = 1
        V[1*p+k1][t_offset+2] = -1
        # C01 += M3, C11 += M3
        W[k1*n+0][t_offset+2] = 1
        W[k1*n+1][t_offset+2] = 1

        # M4: A11, B10-B00
        U[3][t_offset+3] = 1
        V[1*p+k0][t_offset+3] = 1
        V[0*p+k0][t_offset+3] = -1
        # C00 += M4, C10 += M4
        W[k0*n+0][t_offset+3] = 1
        W[k0*n+1][t_offset+3] = 1

        # M5: A00+A01, B11
        U[0][t_offset+4] = 1
        U[1][t_offset+4] = 1
        V[1*p+k1][t_offset+4] = 1
        # C00 -= M5, C01 += M5
        W[k0*n+0][t_offset+4] = -1
        W[k1*n+0][t_offset+4] = 1

        # M6: A10-A00, B00+B01
        U[2][t_offset+5] = 1
        U[0][t_offset+5] = -1
        V[0*p+k0][t_offset+5] = 1
        V[0*p+k1][t_offset+5] = 1
        # C11 += M6
        W[k1*n+1][t_offset+5] = 1

        # M7: A01-A11, B10+B11
        U[1][t_offset+6] = 1
        U[3][t_offset+6] = -1
        V[1*p+k0][t_offset+6] = 1
        V[1*p+k1][t_offset+6] = 1
        # C00 += M7
        W[k0*n+0][t_offset+6] = 1

    # Two Strassen blocks for columns (0,1) and (2,3)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)

    # Schoolbook for column 4
    t = 14
    # C04 = A00*B04 + A01*B14
    U[0][t] = 1
    V[0*p+4][t] = 1
    W[4*n+0][t] = 1
    t += 1
    U[1][t] = 1
    V[1*p+4][t] = 1
    W[4*n+0][t] = 1
    t += 1
    # C14 = A10*B04 + A11*B14
    U[2][t] = 1
    V[0*p+4][t] = 1
    W[4*n+1][t] = 1
    t += 1
    U[3][t] = 1
    V[1*p+4][t] = 1
    W[4*n+1][t] = 1

    return {"U": U, "V": V, "W": W}
