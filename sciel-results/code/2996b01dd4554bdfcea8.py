"""Program for matrix multiplication 4x4x4 using Strassen-based rank-28 decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for 4x4 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    The 4x4 matrices are partitioned into 2x2 blocks of 2x2 submatrices:
    A = [[A00, A01], [A10, A11]]  (each block is 2x2)
    B = [[B00, B01], [B10, B11]]  (each block is 2x2)
    C = [[C00, C01], [C10, C11]]  (each block is 2x2)
    
    Strassen's algorithm computes:
    M1 = (A00+A11) * (B00+B11)
    M2 = (A10+A11) * B00
    M3 = A00 * (B01-B11)
    M4 = A11 * (B10-B00)
    M5 = (A00+A01) * B11
    M6 = (A10-A00) * (B00+B10)
    M7 = (A01-A11) * (B01+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M6 - M7
    
    Each Mi is a 2x2 matrix product, decomposed via schoolbook (rank 4).
    Total rank = 7 * 4 = 28.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Indexing conventions:
    # A[i][j] -> U index: i*m + j = i*4 + j
    # B[j][k] -> V index: j*p + k = j*4 + k
    # C[i][k] -> W index: k*n + i = k*4 + i (per problem statement W[k*n+i, r])
    
    # Helper: add a schoolbook 2x2 matmul term
    # Computes contribution of x*y to C entry (cr*2+ci, cc*2+ck) with coefficient coeff
    # where x is A[ar*2+ai][ac*2+aj] and y is B[br*2+bj][bc*2+bk]
    def add_term(col, ar, ac, ai, aj, br, bc, bj, bk, cr, cc, ci, ck, coeff):
        u_idx = (ar*2+ai)*4 + (ac*2+aj)
        v_idx = (br*2+bj)*4 + (bc*2+bk)
        w_idx = (cc*2+ck)*4 + (cr*2+ci)
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
    
    # For each Strassen multiplication Mk, we compute Xk * Yk where Xk and Yk are
    # linear combinations of 2x2 blocks. We expand each scalar product in the
    # schoolbook decomposition of Xk*Yk.
    
    # Schoolbook 2x2 matmul: if X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X*Y has:
    # z00 = x00*y00 + x01*y10
    # z01 = x00*y01 + x01*y11
    # z10 = x10*y00 + x11*y10
    # z11 = x10*y01 + x11*y11
    
    # For each term x_ij * y_kl, we need to know which A and B elements compose x_ij and y_kl.
    
    col = 0
    
    # M1 = (A00+A11) * (B00+B11)
    # X1 = A00+A11, Y1 = B00+B11
    # X1[0][0] = A00[0][0] + A11[0][0]
    # X1[0][1] = A00[0][1] + A11[0][1]
    # X1[1][0] = A00[1][0] + A11[1][0]
    # X1[1][1] = A00[1][1] + A11[1][1]
    # Y1[0][0] = B00[0][0] + B11[0][0]
    # Y1[0][1] = B00[0][1] + B11[0][1]
    # Y1[1][0] = B00[1][0] + B11[1][0]
    # Y1[1][1] = B00[1][1] + B11[1][1]
    
    # z00 = x00*y00 + x01*y10
    # x00*y00 = (A00[0][0]+A11[0][0])*(B00[0][0]+B11[0][0])
    #   = A00[0][0]*B00[0][0] + A00[0][0]*B11[0][0] + A11[0][0]*B00[0][0] + A11[0][0]*B11[0][0]
    # Each of these contributes to C00[0][0] with coeff +1 (M1 in C00) and C11[0][0] with coeff +1 (M1 in C11)
    
    # For M1, it contributes to C00 with +1 and C11 with +1.
    # So each scalar product in M1 contributes to both C00 and C11 with the same coefficient.
    
    # Let's define a helper that adds all 4 expanded products for a pair of linear forms
    # and applies the C contributions.
    
    def add_m1_term(col, x_a_blocks, y_b_blocks, c_blocks):
        """
        x_a_blocks: list of (block_row, block_col, elem_row, elem_col, coeff) for X element
        y_b_blocks: list of (block_row, block_col, elem_row, elem_col, coeff) for Y element
        c_blocks: list of (block_row, block_col, elem_row, elem_col, coeff) for C contribution
        """
        for (ar, ac, ai, aj, xc) in x_a_blocks:
            for (br, bc, bj, bk, yc) in y_b_blocks:
                for (cr, cc, ci, ck, wc) in c_blocks:
                    add_term(col, ar, ac, ai, aj, br, bc, bj, bk, cr, cc, ci, ck, xc*yc*wc)
    
    # M1 contributes to C00 (+1) and C11 (+1)
    # C00 elements: (0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1)
    # C11 elements: (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)
    
    # M1: z00 = x00*y00 + x01*y10
    # x00 = A00[0][0]+A11[0][0], y00 = B00[0][0]+B11[0][0]
    # x01 = A00[0][1]+A11[0][1], y10 = B00[1][0]+B11[1][0]
    
    # z00 contributes to C00[0][0] and C11[0][0]
    x00 = [(0,0,0,0,1), (1,1,0,0,1)]
    y00 = [(0,0,0,0,1), (1,1,0,0,1)]
    c_z00 = [(0,0,0,0,1), (1,1,0,0,1)]
    add_m1_term(col, x00, y00, c_z00)
    col += 1
    
    x01 = [(0,0,0,1,1), (1,1,0,1,1)]
    y10 = [(0,0,1,0,1), (1,1,1,0,1)]
    add_m1_term(col, x01, y10, c_z00)
    col += 1
    
    # z01 = x00*y01 + x01*y11
    # x01 = B00[0][1]+B11[0][1], y11 = B00[1][1]+B11[1][1]
    # z01 contributes to C00[0][1] and C11[0][1]
    y01 = [(0,0,0,1,1), (1,1,0,1,1)]
    y11 = [(0,0,1,1,1), (1,1,1,1,1)]
    c_z01 = [(0,0,0,1,1), (1,1,0,1,1)]
    add_m1_term(col, x00, y01, c_z01)
    col += 1
    add_m1_term(col, x01, y11, c_z01)
    col += 1
    
    # z10 = x10*y00 + x11*y10
    # x10 = A00[1][0]+A11[1][0], x11 = A00[1][1]+A11[1][1]
    # z10 contributes to C00[1][0] and C11[1][0]
    x10 = [(0,0,1,0,1), (1,1,1,0,1)]
    x11 = [(0,0,1,1,1), (1,1,1,1,1)]
    c_z10 = [(0,0,1,0,1), (1,1,1,0,1)]
    add_m1_term(col, x10, y00, c_z10)
    col += 1
    add_m1_term(col, x11, y10, c_z10)
    col += 1
    
    # z11 = x10*y01 + x11*y11
    # z11 contributes to C00[1][1] and C11[1][1]
    c_z11 = [(0,0,1,1,1), (1,1,1,1,1)]
    add_m1_term(col, x10, y01, c_z11)
    col += 1
    add_m1_term(col, x11, y11, c_z11)
    col += 1
    
    # M2 = (A10+A11) * B00
    # X2 = A10+A11, Y2 = B00
    # X2[0][0] = A10[0][0]+A11[0][0]
    # X2[0][1] = A10[0][1]+A11[0][1]
    # X2[1][0] = A10[1][0]+A11[1][0]
    # X2[1][1] = A10[1][1]+A11[1][1]
    # Y2[0][0] = B00[0][0], Y2[0][1] = B00[0][1], Y2[1][0] = B00[1][0], Y2[1][1] = B00[1][1]
    
    # M2 contributes to C10 with +1
    c_m2 = [(1,0,0,0,1), (1,0,0,1,1), (1,0,1,0,1), (1,0,1,1,1)]
    
    # z00 = x00*y00
    x2_00 = [(1,0,0,0,1), (1,1,0,0,1)]
    y2_00 = [(0,0,0,0,1)]
    c_m2_z00 = [(1,0,0,0,1)]
    add_m1_term(col, x2_00, y2_00, c_m2_z00)
    col += 1
    
    # z01 = x00*y01
    y2_01 = [(0,0,0,1,1)]
    c_m2_z01 = [(1,0,0,1,1)]
    add_m1_term(col, x2_00, y2_01, c_m2_z01)
    col += 1
    
    # z10 = x10*y00
    x2_10 = [(1,0,1,0,1), (1,1,1,0,1)]
    c_m2_z10 = [(1,0,1,0,1)]
    add_m1_term(col, x2_10, y2_00, c_m2_z10)
    col += 1
    
    # z11 = x11*y11
    x2_11 = [(1,0,1,1,1), (1,1,1,1,1)]
    y2_11 = [(0,0,1,1,1)]
    c_m2_z11 = [(1,0,1,1,1)]
    add_m1_term(col, x2_11, y2_11, c_m2_z11)
    col += 1
    
    # M3 = A00 * (B01-B11)
    # X3 = A00, Y3 = B01-B11
    # X3[0][0] = A00[0][0], X3[0][1] = A00[0][1], X3[1][0] = A00[1][0], X3[1][1] = A00[1][1]
    # Y3[0][0] = B01[0][0]-B11[0][0], Y3[0][1] = B01[0][1]-B11[0][1]
    # Y3[1][0] = B01[1][0]-B11[1][0], Y3[1][1] = B01[1][1]-B11[1][1]
    
    # M3 contributes to C01 with +1
    c_m3 = [(0,1,0,0,1), (0,1,0,1,1), (0,1,1,0,1), (0,1,1,1,1)]
    
    x3_00 = [(0,0,0,0,1)]
    x3_01 = [(0,0,0,1,1)]
    x3_10 = [(0,0,1,0,1)]
    x3_11 = [(0,0,1,1,1)]
    
    y3_00 = [(0,1,0,0,1), (1,1,0,0,-1)]
    y3_01 = [(0,1,0,1,1), (1,1,0,1,-1)]
    y3_10 = [(0,1,1,0,1), (1,1,1,0,-1)]
    y3_11 = [(0,1,1,1,1), (1,1,1,1,-1)]
    
    c_m3_z00 = [(0,1,0,0,1)]
    c_m3_z01 = [(0,1,0,1,1)]
    c_m3_z10 = [(0,1,1,0,1)]
    c_m3_z11 = [(0,1,1,1,1)]
    
    add_m1_term(col, x3_00, y3_00, c_m3_z00)
    col += 1
    add_m1_term(col, x3_01, y3_10, c_m3_z00)
    col += 1
    add_m1_term(col, x3_00, y3_01, c_m3_z01)
    col += 1
    add_m1_term(col, x3_01, y3_11, c_m3_z01)
    col += 1
    add_m1_term(col, x3_10, y3_00, c_m3_z10)
    col += 1
    add_m1_term(col, x3_11, y3_10, c_m3_z10)
    col += 1
    add_m1_term(col, x3_10, y3_01, c_m3_z11)
    col += 1
    add_m1_term(col, x3_11, y3_11, c_m3_z11)
    col += 1
    
    # M4 = A11 * (B10-B00)
    # X4 = A11, Y4 = B10-B00
    # X4[0][0] = A11[0][0], X4[0][1] = A11[0][1], X4[1][0] = A11[1][0], X4[1][1] = A11[1][1]
    # Y4[0][0] = B10[0][0]-B00[0][0], Y4[0][1] = B10[0][1]-B00[0][1]
    # Y4[1][0] = B10[1][0]-B00[1][0], Y4[1][1] = B10[1][1]-B00[1][1]
    
    # M4 contributes to C00 with -1 and C10 with +1
    c_m4 = [(0,0,0,0,-1), (0,0,0,1,-1), (0,0,1,0,-1), (0,0,1,1,-1), (1,0,0,0,1), (1,0,0,1,1), (1,0,1,0,1), (1,0,1,1,1)]
    
    x4_00 = [(1,1,0,0,1)]
    x4_01 = [(1,1,0,1,1)]
    x4_10 = [(1,1,1,0,1)]
    x4_11 = [(1,1,1,1,1)]
    
    y4_00 = [(1,0,0,0,1), (0,0,0,0,-1)]
    y4_01 = [(1,0,0,1,1), (0,0,0,1,-1)]
    y4_10 = [(1,0,1,0,1), (0,0,1,0,-1)]
    y4_11 = [(1,0,1,1,1), (0,0,1,1,-1)]
    
    c_m4_z00 = [(0,0,0,0,-1), (1,0,0,0,1)]
    c_m4_z01 = [(0,0,0,1,-1), (1,0,0,1,1)]
    c_m4_z10 = [(0,0,1,0,-1), (1,0,1,0,1)]
    c_m4_z11 = [(0,0,1,1,-1), (1,0,1,1,1)]
    
    add_m1_term(col, x4_00, y4_00, c_m4_z00)
    col += 1
    add_m1_term(col, x4_01, y4_10, c_m4_z00)
    col += 1
    add_m1_term(col, x4_00, y4_01, c_m4_z01)
    col += 1
    add_m1_term(col, x4_01, y4_11, c_m4_z01)
    col += 1
    add_m1_term(col, x4_10, y4_00, c_m4_z10)
    col += 1
    add_m1_term(col, x4_11, y4_10, c_m4_z10)
    col += 1
    add_m1_term(col, x4_10, y4_01, c_m4_z11)
    col += 1
    add_m1_term(col, x4_11, y4_11, c_m4_z11)
    col += 1
    
    # M5 = (A00+A01) * B11
    # X5 = A00+A01, Y5 = B11
    # X5[0][0] = A00[0][0]+A01[0][0], X5[0][1] = A00[0][1]+A01[0][1]
    # X5[1][0] = A00[1][0]+A01[1][0], X5[1][1] = A00[1][1]+A01[1][1]
    # Y5[0][0] = B11[0][0], Y5[0][1] = B11[0][1], Y5[1][0] = B11[1][0], Y5[1][1] = B11[1][1]
    
    # M5 contributes to C00 with +1 and C01 with +1
    c_m5 = [(0,0,0,0,1), (0,0,0,1,1), (0,0,1,0,1), (0,0,1,1,1), (0,1,0,0,1), (0,1,0,1,1), (0,1,1,0,1), (0,1,1,1,1)]
    
    x5_00 = [(0,0,0,0,1), (0,1,0,0,1)]
    x5_01 = [(0,0,0,1,1), (0,1,0,1,1)]
    x5_10 = [(0,0,1,0,1), (0,1,1,0,1)]
    x5_11 = [(0,0,1,1,1), (0,1,1,1,1)]
    
    y5_00 = [(1,1,0,0,1)]
    y5_01 = [(1,1,0,1,1)]
    y5_10 = [(1,1,1,0,1)]
    y5_11 = [(1,1,1,1,1)]
    
    c_m5_z00 = [(0,0,0,0,1), (0,1,0,0,1)]
    c_m5_z01 = [(0,0,0,1,1), (0,1,0,1,1)]
    c_m5_z10 = [(0,0,1,0,1), (0,1,1,0,1)]
    c_m5_z11 = [(0,0,1,1,1), (0,1,1,1,1)]
    
    add_m1_term(col, x5_00, y5_00, c_m5_z00)
    col += 1
    add_m1_term(col, x5_01, y5_10, c_m5_z00)
    col += 1
    add_m1_term(col, x5_00, y5_01, c_m5_z01)
    col += 1
    add_m1_term(col, x5_01, y5_11, c_m5_z01)
    col += 1
    add_m1_term(col, x5_10, y5_00, c_m5_z10)
    col += 1
    add_m1_term(col, x5_11, y5_10, c_m5_z10)
    col += 1
    add_m1_term(col, x5_10, y5_01, c_m5_z11)
    col += 1
    add_m1_term(col, x5_11, y5_11, c_m5_z11)
    col += 1
    
    # M6 = (A10-A00) * (B00+B10)
    # X6 = A10-A00, Y6 = B00+B10
    # X6[0][0] = A10[0][0]-A00[0][0], X6[0][1] = A10[0][1]-A00[0][1]
    # X6[1][0] = A10[1][0]-A00[1][0], X6[1][1] = A10[1][1]-A00[1][1]
    # Y6[0][0] = B00[0][0]+B10[0][0], Y6[0][1] = B00[0][1]+B10[0][1]
    # Y6[1][0] = B00[1][0]+B10[1][0], Y6[1][1] = B00[1][1]+B10[1][1]
    
    # M6 contributes to C11 with +1
    c_m6 = [(1,1,0,0,1), (1,1,0,1,1), (1,1,1,0,1), (1,1,1,1,1)]
    
    x6_00 = [(1,0,0,0,1), (0,0,0,0,-1)]
    x6_01 = [(1,0,0,1,1), (0,0,0,1,-1)]
    x6_10 = [(1,0,1,0,1), (0,0,1,0,-1)]
    x6_11 = [(1,0,1,1,1), (0,0,1,1,-1)]
    
    y6_00 = [(0,0,0,0,1), (1,0,0,0,1)]
    y6_01 = [(0,0,0,1,1), (1,0,0,1,1)]
    y6_10 = [(0,0,1,0,1), (1,0,1,0,1)]
    y6_11 = [(0,0,1,1,1), (1,0,1,1,1)]
    
    c_m6_z00 = [(1,1,0,0,1)]
    c_m6_z01 = [(1,1,0,1,1)]
    c_m6_z10 = [(1,1,1,0,1)]
    c_m6_z11 = [(1,1,1,1,1)]
    
    add_m1_term(col, x6_00, y6_00, c_m6_z00)
    col += 1
    add_m1_term(col, x6_01, y6_10, c_m6_z00)
    col += 1
    add_m1_term(col, x6_00, y6_01, c_m6_z01)
    col += 1
    add_m1_term(col, x6_01, y6_11, c_m6_z01)
    col += 1
    add_m1_term(col, x6_10, y6_00, c_m6_z10)
    col += 1
    add_m1_term(col, x6_11, y6_10, c_m6_z10)
    col += 1
    add_m1_term(col, x6_10, y6_01, c_m6_z11)
    col += 1
    add_m1_term(col, x6_11, y6_11, c_m6_z11)
    col += 1
    
    # M7 = (A01-A11) * (B01+B11)
    # X7 = A01-A11, Y7 = B01+B11
    # X7[0][0] = A01[0][0]-A11[0][0], X7[0][1] = A01[0][1]-A11[0][1]
    # X7[1][0] = A01[1][0]-A11[1][0], X7[1][1] = A01[1][1]-A11[1][1]
    # Y7[0][0] = B01[0][0]+B11[0][0], Y7[0][1] = B01[0][1]+B11[0][1]
    # Y7[1][0] = B01[1][0]+B11[1][0], Y7[1][1] = B01[1][1]+B11[1][1]
    
    # M7 contributes to C00 with -1 and C11 with -1
    c_m7 = [(0,0,0,0,-1), (0,0,0,1,-1), (0,0,1,0,-1), (0,0,1,1,-1), (1,1,0,0,-1), (1,1,0,1,-1), (1,1,1,0,-1), (1,1,1,1,-1)]
    
    x7_00 = [(0,1,0,0,1), (1,1,0,0,-1)]
    x7_01 = [(0,1,0,1,1), (1,1,0,1,-1)]
    x7_10 = [(0,1,1,0,1), (1,1,1,0,-1)]
    x7_11 = [(0,1,1,1,1), (1,1,1,1,-1)]
    
    y7_00 = [(0,1,0,0,1), (1,1,0,0,1)]
    y7_01 = [(0,1,0,1,1), (1,1,0,1,1)]
    y7_10 = [(0,1,1,0,1), (1,1,1,0,1)]
    y7_11 = [(0,1,1,1,1), (1,1,1,1,1)]
    
    c_m7_z00 = [(0,0,0,0,-1), (1,1,0,0,-1)]
    c_m7_z01 = [(0,0,0,1,-1), (1,1,0,1,-1)]
    c_m7_z10 = [(0,0,1,0,-1), (1,1,1,0,-1)]
    c_m7_z11 = [(0,0,1,1,-1), (1,1,1,1,-1)]
    
    add_m1_term(col, x7_00, y7_00, c_m7_z00)
    col += 1
    add_m1_term(col, x7_01, y7_10, c_m7_z00)
    col += 1
    add_m1_term(col, x7_00, y7_01, c_m7_z01)
    col += 1
    add_m1_term(col, x7_01, y7_11, c_m7_z01)
    col += 1
    add_m1_term(col, x7_10, y7_00, c_m7_z10)
    col += 1
    add_m1_term(col, x7_11, y7_10, c_m7_z10)
    col += 1
    add_m1_term(col, x7_10, y7_01, c_m7_z11)
    col += 1
    add_m1_term(col, x7_11, y7_11, c_m7_z11)
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
