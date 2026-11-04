import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS  # 5, 5, 5

    # Schoolbook decomposition as fallback
    def schoolbook_factors(n, m, p):
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
        return U, V, W

    # Strassen decomposition for 2x2x2 (rank 7)
    # Factors are: U[2*2, 7], V[2*2, 7], W[2*2, 7]
    # Using the standard Strassen algorithm factors
    def strassen_factors():
        # U: rows indexed by (i, j) where i in {0,1}, j in {0,1}
        # V: rows indexed by (j, k) where j in {0,1}, k in {0,1}
        # W: rows indexed by (k, i) where k in {0,1}, i in {0,1}
        #
        # The 7 multiplications in Strassen are:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # U[i*m+j, r] gives coefficient for A_{i,j} in multiplication r
        # V[j*p+k, r] gives coefficient for B_{j,k} in multiplication r
        # W[k*n+i, r] gives coefficient for C_{i,k} from multiplication r

        U = [[0] * 7 for _ in range(4)]  # n*m = 4
        V = [[0] * 7 for _ in range(4)]  # m*p = 4
        W = [[0] * 7 for _ in range(4)]  # p*n = 4

        # Row indices:
        # U: row 0 = A[0,0], row 1 = A[0,1], row 2 = A[1,0], row 3 = A[1,1]
        # V: row 0 = B[0,0], row 1 = B[0,1], row 2 = B[1,0], row 3 = B[1,1]
        # W: row 0 = C[0,0] (k=0,i=0), row 1 = C[0,1] (k=0,i=1),
        #    row 2 = C[1,0] (k=1,i=0), row 3 = C[1,1] (k=1,i=1)

        # M1 = (A00 + A11) * (B00 + B11)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        W[0][0] = 1  # C00 += M1
        W[3][0] = 1  # C11 += M1

        # M2 = (A10 + A11) * B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10 += M2
        W[3][1] = -1  # C11 -= M2

        # M3 = A00 * (B01 - B11)
        U[0][2] = 1  # A00
        V[1][2] = 1  # B01
        V[3][2] = -1  # B11
        W[1][2] = 1  # C01 += M3
        W[3][2] = 1  # C11 += M3

        # M4 = A11 * (B10 - B00)
        U[3][3] = 1  # A11
        V[2][3] = 1  # B10
        V[0][3] = -1  # B00
        W[0][3] = 1  # C00 += M4
        W[2][3] = 1  # C10 += M4

        # M5 = (A00 + A01) * B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        W[0][4] = -1  # C00 -= M5
        W[1][4] = 1  # C01 += M5

        # M6 = (A10 - A00) * (B00 + B01)
        U[2][5] = 1  # A10
        U[0][5] = -1  # A00
        V[0][5] = 1  # B00
        V[1][5] = 1  # B01
        W[3][5] = 1  # C11 += M6

        # M7 = (A01 - A11) * (B10 + B11)
        U[1][6] = 1  # A01
        U[3][6] = -1  # A11
        V[2][6] = 1  # B10
        V[3][6] = 1  # B11
        W[0][6] = 1  # C00 += M7

        return U, V, W

    # Try 2+2+1 partition with Strassen for 2x2 blocks
    # Partition sizes for rows of A (n=5): [2, 2, 1]
    # Partition sizes for cols of A / rows of B (m=5): [2, 2, 1]
    # Partition sizes for cols of B (p=5): [2, 2, 1]

    row_part = [2, 2, 1]  # partitions of n=5
    col_part = [2, 2, 1]  # partitions of m=5
    out_part = [2, 2, 1]  # partitions of p=5

    # Calculate total rank
    total_rank = 0
    for ri in range(3):
        for cj in range(3):
            for ok in range(3):
                ni = row_part[ri]
                mj = col_part[cj]
                pk = out_part[ok]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += ni * mj * pk  # Schoolbook

    # Build global factor matrices
    U_global = [[0] * total_rank for _ in range(n * m)]  # 25 x total_rank
    V_global = [[0] * total_rank for _ in range(m * p)]  # 25 x total_rank
    W_global = [[0] * total_rank for _ in range(p * n)]  # 25 x total_rank

    col_offset = 0
    row_start = 0
    for ri in range(3):
        row_end = row_start + row_part[ri]
        ni = row_part[ri]
        col_start_a = 0
        for cj in range(3):
            col_end_a = col_start_a + col_part[cj]
            mj = col_part[cj]
            out_start_b = 0
            for ok in range(3):
                out_end_b = out_start_b + out_part[ok]
                pk = out_part[ok]

                if time.time() > deadline:
                    # Fallback to schoolbook
                    return {"U": schoolbook_factors(n, m, p)[0],
                            "V": schoolbook_factors(n, m, p)[1],
                            "W": schoolbook_factors(n, m, p)[2]}

                if ni == 2 and mj == 2 and pk == 2:
                    U_local, V_local, W_local = strassen_factors()
                    local_rank = 7
                else:
                    U_local, V_local, W_local = schoolbook_factors(ni, mj, pk)
                    local_rank = ni * mj * pk

                # Map local factors to global
                for i in range(ni):
                    for j in range(mj):
                        u_row = (row_start + i) * m + (col_start_a + j)
                        for r in range(local_rank):
                            U_global[u_row][col_offset + r] = U_local[i * mj + j][r]

                for j in range(mj):
                    for k in range(pk):
                        v_row = (col_start_a + j) * p + (out_start_b + k)
                        for r in range(local_rank):
                            V_global[v_row][col_offset + r] = V_local[j * pk + k][r]

                for k in range(pk):
                    for i in range(ni):
                        w_row = (out_start_b + k) * n + (row_start + i)
                        for r in range(local_rank):
                            W_global[w_row][col_offset + r] = W_local[k * ni + i][r]

                col_offset += local_rank
                out_start_b = out_end_b
            col_start_a = col_end_a
        row_start = row_end

    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END
