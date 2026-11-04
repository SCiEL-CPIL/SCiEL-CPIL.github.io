DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(M, r, c, val):
        M[r][c] += val

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1 (row 0), 3,4 (row 1)
    # B indices: 0,1 (col 0), 3,4 (col 1) -- wait, B is 3x3.
    # B[j, k]. j in 0,1. k in 0,1.
    # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4.
    # C indices: k*n+i.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3.

    # m1 = (A00+A11)(B00+B11)
    # A00=0, A11=4. B00=0, B11=4.
    set_val(U, 0, 0, 1); set_val(U, 4, 0, 1)
    set_val(V, 0, 0, 1); set_val(V, 4, 0, 1)
    set_val(W, 0, 0, 1); set_val(W, 1, 0, 1); set_val(W, 2, 0, 1); set_val(W, 3, 0, 1)

    # m2 = (A10+A11)B00
    # A10=3, A11=4. B00=0.
    set_val(U, 3, 1, 1); set_val(U, 4, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 2, 1, 1); set_val(W, 3, 1, 1)

    # m3 = A00(B01-B11)
    # A00=0. B01=1, B11=4.
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1); set_val(V, 4, 2, -1)
    set_val(W, 1, 2, 1); set_val(W, 3, 2, 1)

    # m4 = A11(B10-B00)
    # A11=4. B10=3, B00=0.
    set_val(U, 4, 3, 1)
    set_val(V, 3, 3, 1); set_val(V, 0, 3, -1)
    set_val(W, 2, 3, 1); set_val(W, 3, 3, 1)

    # m5 = (A00+A01)B11
    # A00=0, A01=1. B11=4.
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    set_val(V, 4, 4, 1)
    set_val(W, 0, 4, 1); set_val(W, 1, 4, 1)

    # m6 = (A10-A00)(B00+B01)
    # A10=3, A00=0. B00=0, B01=1.
    set_val(U, 3, 5, 1); set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1); set_val(V, 1, 5, 1)
    set_val(W, 3, 5, 1)

    # m7 = (A01-A11)(B10+B11)
    # A01=1, A11=4. B10=3, B11=4.
    set_val(U, 1, 6, 1); set_val(U, 4, 6, -1)
    set_val(V, 3, 6, 1); set_val(V, 4, 6, 1)
    set_val(W, 0, 6, 1); set_val(W, 2, 6, 1)

    # 2. Schoolbook for A[0:2, 0:2] * B[0:2, 2:3]
    # A indices: 0,1,3,4.
    # B indices: B[0,2]=2, B[1,2]=5.
    # C indices: C[0,2]=4, C[1,2]=5.

    # A00 B02 -> C02
    set_val(U, 0, 7, 1)
    set_val(V, 2, 7, 1)
    set_val(W, 4, 7, 1)

    # A01 B12 -> C02
    set_val(U, 1, 8, 1)
    set_val(V, 5, 8, 1)
    set_val(W, 4, 8, 1)

    # A10 B02 -> C12
    set_val(U, 3, 9, 1)
    set_val(V, 2, 9, 1)
    set_val(W, 5, 9, 1)

    # A11 B12 -> C12
    set_val(U, 4, 10, 1)
    set_val(V, 5, 10, 1)
    set_val(W, 5, 10, 1)

    # 3. Schoolbook for A[0:2, 2:3] * B[2:3, 0:3]
    # A indices: A[0,2]=2, A[1,2]=5.
    # B indices: B[2,0]=6, B[2,1]=7, B[2,2]=8.
    # C indices: C[0,0]=0, C[0,1]=1, C[0,2]=4, C[1,0]=2, C[1,1]=3, C[1,2]=5.

    # A02 B20 -> C00
    set_val(U, 2, 11, 1)
    set_val(V, 6, 11, 1)
    set_val(W, 0, 11, 1)

    # A02 B21 -> C01
    set_val(U, 2, 12, 1)
    set_val(V, 7, 12, 1)
    set_val(W, 1, 12, 1)

    # A02 B22 -> C02
    set_val(U, 2, 13, 1)
    set_val(V, 8, 13, 1)
    set_val(W, 4, 13, 1)

    # A12 B20 -> C10
    set_val(U, 5, 14, 1)
    set_val(V, 6, 14, 1)
    set_val(W, 2, 14, 1)

    # A12 B21 -> C11
    set_val(U, 5, 15, 1)
    set_val(V, 7, 15, 1)
    set_val(W, 3, 15, 1)

    # A12 B22 -> C12
    set_val(U, 5, 16, 1)
    set_val(V, 8, 16, 1)
    set_val(W, 5, 16, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
