"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns Strassen's rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    Convention:
        U[i*m + j, c] for A[i, j]
        V[j*p + k, c] for B[j, k]
        W[k*n + i, c] for C[i, k]  (official trace convention)
    
    with n = m = p = 2, rank = 7.
    
    The decomposition satisfies:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for all valid (i,j,k) triples.
    """
    n, m, p = 2, 2, 2
    rank = 7

    # U rows: A00=0, A01=1, A10=2, A11=3  (index = i*m + j)
    # V rows: B00=0, B01=1, B10=2, B11=3  (index = j*p + k)
    # W rows: C00=0, C10=1, C01=2, C11=3  (index = k*n + i)

    # Strassen's 7 products:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    #
    # Output:
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5

    U = [
        [1,  0, 1,  0, 1, -1,  0],  # A00
        [0,  0, 0,  0, 1,  0,  1],  # A01
        [0,  1, 0,  0, 0,  1,  0],  # A10
        [1,  1, 0,  1, 0,  0, -1],  # A11
    ]
    V = [
        [1,  1, 0, -1, 0,  1,  0],  # B00
        [0,  0, 1,  0, 0,  1,  0],  # B01
        [0,  0, 0,  1, 0,  0,  1],  # B10
        [1,  0, -1, 0, 1,  0,  1],  # B11
    ]
    W = [
        [1,  0, 0, 1, -1, 0, 1],  # C00 (k=0,i=0)
        [0,  1, 0, 1,  0, 0, 0],  # C10 (k=0,i=1)
        [0,  0, 1, 0,  1, 0, 0],  # C01 (k=1,i=0)
        [1, -1, 1, 0,  0, 1, 0],  # C11 (k=1,i=1)
    ]

    # Verify exactness: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    """Fallback: rank-8 trivial decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
