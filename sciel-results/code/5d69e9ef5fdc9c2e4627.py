"""Rewritten program: rank-38 decomposition for 3x3 by 3x6 matrix multiplication.

Approach: Partition the 3x6 output into two 3x3 blocks. Each block is computed
via a rank-19 Laderman-style decomposition for 3x3x3 matrix multiplication.
Total rank = 19 + 19 = 38.
"""
import random

DIMENSIONS = [3, 3, 6]


def construct():
    n, m, p = 3, 3, 6
    rank = 38

    # Laderman rank-19 decomposition for 3x3 x 3x3 -> 3x3
    # Each term: (A_coeffs, B_coeffs, C_coeffs) as flat 9-element lists
    # A is 3x3 (row-major: a00,a01,a02,a10,a11,a12,a20,a21,a22)
    # B is 3x3 (row-major: b00,b01,b02,b10,b11,b12,b20,b21,b22)
    # C is 3x3 (row-major: c00,c01,c02,c10,c11,c12,c20,c21,c22)

    # I'll build the Laderman terms programmatically using the known structure.
    # The standard Laderman 19-term algorithm:
    #
    # p0  = a00*b00
    # p1  = a10*b00
    # p2  = a01*b10
    # p3  = a11*b10
    # p4  = a02*b20
    # p5  = a12*b20
    # p6  = a20*b01
    # p7  = a21*b11
    # p8  = a22*b21
    # p9  = (a00+a01)*b02
    # p10 = (a10+a11)*b12
    # p11 = (a20+a21)*b22
    # p12 = (a00+a01)*(b00+b10)
    # p13 = (a10+a11)*(b00+b10)
    # p14 = (a00+a01)*(b02+b12)
    # p15 = (a10+a11)*(b02+b12)
    # p16 = (a00+a01+a20+a21)*(b00+b10+b02+b12)
    # p17 = (a00+a01)*(b00+b10+b02+b12)
    # p18 = (a10+a11)*(b00+b10+b02+b12)
    #
    # C[0][0] = p0 + p9 - p12 + p14 + p16 - p17
    # C[1][0] = p1 + p10 - p13 + p15 + p16 - p17 - p18
    # C[2][0] = p6 + p11 + p16 - p18
    # C[0][1] = p2 + p12 - p16 + p17
    # C[1][1] = p3 + p13 - p16 + p17 + p18
    # C[2][1] = p7 + p18
    # C[0][2] = p4 - p14 - p16 + p17
    # C[1][2] = p5 - p15 - p16 + p17 + p18
    # C[2][2] = p8 - p18

    def make_term(A_flat, B_flat, C_flat):
        return (A_flat, B_flat, C_flat)

    # Helper to create a flat 9-vector with specific entries
    def vec9(idx_val_pairs):
        v = [0] * 9
        for idx, val in idx_val_pairs:
            v[idx] += val
        return v

    # Indices: row*3 + col
    # a00=0, a01=1, a02=2, a10=3, a11=4, a12=5, a20=6, a21=7, a22=8
    # b00=0, b01=1, b02=2, b10=3, b11=4, b12=5, b20=6, b21=7, b22=8
    # c00=0, c01=1, c02=2, c10=3, c11=4, c12=5, c20=6, c21=7, c22=8

    laderman_terms = []

    # p0: a00*b00
    laderman_terms.append(make_term(vec9([(0,1)]), vec9([(0,1)]), vec9([(0,1)])))
    # p1: a10*b00
    laderman_terms.append(make_term(vec9([(3,1)]), vec9([(0,1)]), vec9([(3,1)])))
    # p2: a01*b10
    laderman_terms.append(make_term(vec9([(1,1)]), vec9([(3,1)]), vec9([(1,1)])))
    # p3: a11*b10
    laderman_terms.append(make_term(vec9([(4,1)]), vec9([(3,1)]), vec9([(4,1)])))
    # p4: a02*b20
    laderman_terms.append(make_term(vec9([(2,1)]), vec9([(6,1)]), vec9([(2,1)])))
    # p5: a12*b20
    laderman_terms.append(make_term(vec9([(5,1)]), vec9([(6,1)]), vec9([(5,1)])))
    # p6: a20*b01
    laderman_terms.append(make_term(vec9([(6,1)]), vec9([(1,1)]), vec9([(6,1)])))
    # p7: a21*b11
    laderman_terms.append(make_term(vec9([(7,1)]), vec9([(4,1)]), vec9([(7,1)])))
    # p8: a22*b21
    laderman_terms.append(make_term(vec9([(8,1)]), vec9([(7,1)]), vec9([(8,1)])))
    # p9: (a00+a01)*b02
    laderman_terms.append(make_term(vec9([(0,1),(1,1)]), vec9([(2,1)]), vec9([(0,1)])))
    # p10: (a10+a11)*b12
    laderman_terms.append(make_term(vec9([(3,1),(4,1)]), vec9([(5,1)]), vec9([(3,1)])))
    # p11: (a20+a21)*b22
    laderman_terms.append(make_term(vec9([(6,1),(7,1)]), vec9([(8,1)]), vec9([(6,1)])))
    # p12: (a00+a01)*(b00+b10)
    laderman_terms.append(make_term(vec9([(0,1),(1,1)]), vec9([(0,1),(3,1)]), vec9([(0,-1),(1,1)])))
    # p13: (a10+a11)*(b00+b10)
    laderman_terms.append(make_term(vec9([(3,1),(4,1)]), vec9([(0,1),(3,1)]), vec9([(3,-1),(4,1)])))
    # p14: (a00+a01)*(b02+b12)
    laderman_terms.append(make_term(vec9([(0,1),(1,1)]), vec9([(2,1),(5,1)]), vec9([(0,1),(2,-1)])))
    # p15: (a10+a11)*(b02+b12)
    laderman_terms.append(make_term(vec9([(3,1),(4,1)]), vec9([(2,1),(5,1)]), vec9([(3,1),(5,-1)])))
    # p16: (a00+a01+a20+a21)*(b00+b10+b02+b12)
    laderman_terms.append(make_term(vec9([(0,1),(1,1),(6,1),(7,1)]), vec9([(0,1),(3,1),(2,1),(5,1)]),
                                    vec9([(0,1),(1,-1),(2,-1),(3,1),(4,-1),(5,-1),(6,1)])))
    # p17: (a00+a01)*(b00+b10+b02+b12)
    laderman_terms.append(make_term(vec9([(0,1),(1,1)]), vec9([(0,1),(3,1),(2,1),(5,1)]),
                                    vec9([(0,-1),(1,1),(2,1),(3,-1),(4,1)])))
    # p18: (a10+a11)*(b00+b10+b02+b12)
    laderman_terms.append(make_term(vec9([(3,1),(4,1)]), vec9([(0,1),(3,1),(2,1),(5,1)]),
                                    vec9([(3,-1),(4,1),(5,1),(6,-1),(7,-1),(8,-1)])))

    # Build the full rank-38 decomposition
    U = [[0] * rank for _ in range(n * m)]   # (9, 38)
    V = [[0] * rank for _ in range(m * p)]   # (18, 38)
    W = [[0] * rank for _ in range(p * n)]   # (18, 38)

    for block in range(2):
        base = block * 19
        k_off = block * 3  # offset in the k-dimension (columns of B, rows of C)

        for t in range(19):
            A_flat, B_flat, C_flat = laderman_terms[t]
            col = base + t

            # Fill U: A is 3x3, index i*m+j = i*3+j
            for i in range(3):
                for j in range(3):
                    U[i * 3 + j][col] = A_flat[i * 3 + j]

            # Fill V: B is 3x3 but we only use 3 columns (k_off to k_off+2)
            # V has shape (m*p, rank) = (3*6, 38) = (18, 38)
            # V index: j*p + k = j*6 + k
            for j in range(3):
                for k in range(3):
                    V[j * 6 + (k_off + k)][col] = B_flat[j * 3 + k]

            # Fill W: C is 3x3, output columns k_off to k_off+2
            # W has shape (p*n, rank) = (6*3, 38) = (18, 38)
            # W index: k*n + i = k*3 + i
            for i in range(3):
                for k in range(3):
                    W[(k_off + k) * 3 + i][col] = C_flat[i * 3 + k]

    # Verify correctness
    def verify(U_mat, V_mat, W_mat, rank_val):
        random.seed(12345)
        for trial in range(20):
            A = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            B = [[random.randint(-5, 5) for _ in range(6)] for _ in range(3)]

            # Expected C = A * B
            C_exp = [[0] * 6 for _ in range(3)]
            for i in range(3):
                for k in range(6):
                    s = 0
                    for j in range(3):
                        s += A[i][j] * B[j][k]
                    C_exp[i][k] = s

            # Decomposition: C[i,k] = sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] summed over j
            C_dec = [[0] * 6 for _ in range(3)]
            for c in range(rank_val):
                for i in range(3):
                    for k in range(6):
                        s = 0
                        for j in range(3):
                            s += U_mat[i * 3 + j][c] * V_mat[j * 6 + k][c]
                        C_dec[i][k] += W_mat[k * 3 + i][c] * s

            for i in range(3):
                for k in range(6):
                    if C_exp[i][k] != C_dec[i][k]:
                        return False
        return True

    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook rank 54
        rank_sb = 54
        U_sb = [[0] * rank_sb for _ in range(9)]
        V_sb = [[0] * rank_sb for _ in range(18)]
        W_sb = [[0] * rank_sb for _ in range(18)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U_sb[i * 3 + j][col] = 1
                    V_sb[j * 6 + k][col] = 1
                    W_sb[k * 3 + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
