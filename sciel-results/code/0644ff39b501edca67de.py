def construct():
    n, m, p = 2, 5, 6
    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_2x2_2x2(a_col_offset, b_row_offset, b_col_offset):
        nonlocal col

        def set_u(i, j, val):
            U[i * m + a_col_offset + j][col] += val

        def set_v(j, k, val):
            V[(b_row_offset + j) * p + b_col_offset + k][col] += val

        def set_w(k, i, val):
            W[(b_col_offset + k) * n + i][col] += val

        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0]+=1, C[1,1]+=1
        set_u(0, 0, 1); set_u(1, 1, 1)
        set_v(0, 0, 1); set_v(1, 1, 1)
        set_w(0, 0, 1); set_w(1, 1, 1)
        col += 1

        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0]+=1, C[1,1]+=1
        set_u(1, 0, 1); set_u(1, 1, 1)
        set_v(0, 0, 1)
        set_w(0, 1, 1); set_w(1, 1, 1)
        col += 1

        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1]+=1, C[1,1]+=1
        set_u(0, 0, 1)
        set_v(0, 1, 1); set_v(1, 1, -1)
        set_w(1, 0, 1); set_w(1, 1, 1)
        col += 1

        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0]+=1, C[1,0]+=1
        set_u(1, 1, 1)
        set_v(1, 0, 1); set_v(0, 0, -1)
        set_w(0, 0, 1); set_w(0, 1, 1)
        col += 1

        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0]+=-1, C[0,1]+=+1
        set_u(0, 0, 1); set_u(0, 1, 1)
        set_v(1, 1, 1)
        set_w(0, 0, -1); set_w(1, 0, 1)
        col += 1

        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1]+=+1
        set_u(1, 0, 1); set_u(0, 0, -1)
        set_v(0, 0, 1); set_v(0, 1, 1)
        set_w(1, 1, 1)
        col += 1

        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0]+=+1
        set_u(0, 1, 1); set_u(1, 1, -1)
        set_v(1, 0, 1); set_v(1, 1, 1)
        set_w(0, 0, 1)
        col += 1

    # 6 Strassen blocks (2 A-blocks x 3 B-column-blocks)
    for a_col in [0, 2]:
        for b_row in [0, 2]:
            for b_col in [0, 2, 4]:
                add_strassen_2x2_2x2(a_col, b_row, b_col)

    # Last term: A[:,4] * B[4,:]
    for i in range(2):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}
