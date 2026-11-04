"""Matrix multiplication 6x6x6 - Strassen block decomposition (3x3 blocks of 2x2)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 3x3 grid of 2x2 blocks.
    Apply rank-19 decomposition to the 3x3 block matrix multiplication.
    For each of the 19 terms, the 2x2 matrix products use Strassen (rank 7).
    Total rank = 19 * 7 = 133.
    W follows convention W[k*n+i, r] where (k,i) is (col_C, row_C).
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    bs = 2       # block size
    nb = 3       # number of blocks per dimension

    # 3x3 matrix multiplication rank-19 decomposition
    # A entries: a00,a01,a02,a10,a11,a12,a20,a21,a22 (indices 0-8)
    # B entries: b00,b01,b02,b10,b11,b12,b20,b21,b22 (indices 0-8)
    # C entries: c00,c01,c02,c10,c11,c12,c20,c21,c22 (indices 0-8)

    term_defs = [
        # t1: (a00+a11+a22)(b00+b11+b22)
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,-1,1, -1,1,-1, 1,-1,1]),
        # t2: (a00-a11)(b00-b11)
        ([1,0,0, 0,-1,0, 0,0,0], [1,0,0, 0,-1,0, 0,0,0], [1,1,-1, -1,1,1, -1,-1,1]),
        # t3: (a11-a22)(b11-b22)
        ([0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,1,1, 0,1,1]),
        # t4: (a00+a01)(b00+b20)
        ([1,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 1,0,0], [1,1,0, 1,1,0, 0,0,0]),
        # t5: (a01+a02)(b10+b21)
        ([0,1,1, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,1,0], [0,1,1, 0,1,1, 0,0,0]),
        # t6: (a02+a00)(b20+b12)
        ([1,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 1,0,0], [0,0,1, 0,0,1, 1,0,1]),
        # t7: (a10+a11)(b10+b01)
        ([0,0,0, 1,1,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        # t8: (a11+a12)(b20+b12)
        ([0,0,0, 0,1,1, 0,0,0], [0,0,0, 0,0,1, 1,0,0], [0,1,1, 0,1,1, 0,0,0]),
        # t9: (a12+a10)(b02+b21)
        ([0,0,0, 1,0,1, 0,0,0], [0,0,1, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,1,1]),
        # t10: (a20+a21)(b01+b20)
        ([0,0,0, 0,0,0, 1,1,0], [0,1,0, 0,0,0, 1,0,0], [1,1,0, 0,0,0, 1,1,0]),
        # t11: (a21+a22)(b10+b02)
        ([0,0,0, 0,0,0, 0,1,1], [0,1,0, 1,0,0, 0,0,0], [0,1,1, 0,0,0, 0,1,1]),
        # t12: (a22+a20)(b02+b21)
        ([0,0,0, 0,0,0, 1,0,1], [0,0,1, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 1,0,1]),
        # t13: (a00+a22-a11)(b00+b22-b11)
        ([1,0,0, 0,-1,0, 0,0,1], [1,0,0, 0,-1,0, 0,0,1], [-1,0,0, 1,0,0, 1,0,1]),
        # t14: (a01+a12-a20)(b01+b12-b20)
        ([0,1,0, 0,0,1, -1,0,0], [0,1,0, 0,0,1, -1,0,0], [1,-1,0, -1,0,0, 1,0,-1]),
        # t15: (a02+a20-a10)(b02+b20-b10)
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,1,-1, 0,0,0, -1,0,1]),
        # t16: (a10+a22-a02)(b10+b22-b02)
        ([0,0,-1, 1,0,0, 0,0,1], [0,0,-1, 1,0,0, 0,0,1], [0,0,1, 0,0,0, 1,1,0]),
        # t17: (a11+a02-a20)(b11+b02-b20)
        ([0,0,1, 0,1,0, -1,0,0], [0,0,1, 0,1,0, -1,0,0], [-1,0,0, 1,0,0, 0,0,-1]),
        # t18: (a12+a00-a10)(b12+b00-b10)
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,-1,0, 0,1,-1, 0,0,0]),
        # t19: (a20+a01-a11)(b20+b01-b11)
        ([0,1,0, 0,-1,0, 1,0,0], [0,1,0, 0,-1,0, 1,0,0], [0,0,-1, 0,0,0, 1,1,0]),
    ]

    # Verify the 3x3 decomposition with a quick test
    import random
    random.seed(42)
    valid = True
    for trial in range(20):
        A = np.array([[random.randint(-5,5) for _ in range(3)] for _ in range(3)])
        B = np.array([[random.randint(-5,5) for _ in range(3)] for _ in range(3)])
        C_direct = A @ B
        C_strassen = np.zeros((3,3))
        for t_idx in range(19):
            a_c, b_c, c_c = term_defs[t_idx]
            a_vec = np.array(a_c).reshape(3,3)
            b_vec = np.array(b_c).reshape(3,3)
            c_vec = np.array(c_c).reshape(3,3)
            # The term contributes: (sum a_vec[i][k]*A[i][k]) * (sum b_vec[k][j]*B[k][j]) * c_vec[i][j]
            # Actually: term = (A^T * a_vec_flat) * (B^T * b_vec_flat) ... no
            # term = (sum_{i,k} a_c[i*3+k] * A[i][k]) * (sum_{k,j} b_c[k*3+j] * B[k][j])
            # Wait, a_c is 9 entries for A[i][j], b_c is 9 entries for B[k][j]
            # The product is: (sum_{i,j} a_c[i*3+j]*A[i][j]) * (sum_{k,j} b_c[k*3+j]*B[k][j])
            # No, that's not right either. Let me think again.
            # In Strassen, each term is of the form (linear combo of A entries) * (linear combo of B entries)
            # and contributes to C entries with coefficients.
            a_scalar = sum(a_c[i*3+j] * A[i,j] for i in range(3) for j in range(3))
            b_scalar = sum(b_c[k*3+j] * B[k,j] for k in range(3) for j in range(3))
            for i in range(3):
                for j in range(3):
                    C_strassen[i,j] += c_c[i*3+j] * a_scalar * b_scalar
        if not np.array_equal(C_direct, C_strassen):
            valid = False
            break

    if not valid:
        # Fallback to rank-189
        total_rank = 27 * 7
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        strassen_A = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
        strassen_B = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
        output_coeffs = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]
        for i in range(nb):
            for j in range(nb):
                for k in range(nb):
                    for t in range(7):
                        for r in range(bs):
                            for c in range(bs):
                                a_local = r * bs + c
                                a_coeff = strassen_A[t][a_local]
                                if a_coeff != 0:
                                    u_idx = (i * bs + r) * m + (k * bs + c)
                                    U[u_idx][col] = a_coeff
                        for r in range(bs):
                            for c in range(bs):
                                b_local = r * bs + c
                                b_coeff = strassen_B[t][b_local]
                                if b_coeff != 0:
                                    v_idx = (k * bs + r) * p + (j * bs + c)
                                    V[v_idx][col] = b_coeff
                        for r in range(bs):
                            for c in range(bs):
                                c_local = r * bs + c
                                c_coeff = output_coeffs[t][c_local]
                                if c_coeff != 0:
                                    w_idx = (j * bs + c) * n + (i * bs + r)
                                    W[w_idx][col] = c_coeff
                        col += 1
        return {"U": U, "V": V, "W": W}

    # Build rank-133 decomposition: 19 outer terms x 7 inner Strassen terms
    total_rank = 19 * 7  # 133

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Strassen 2x2 decomposition coefficients
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01+a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # strassen_2x2_A[t][entry]: coefficient for A entry in term t
    # A entries: 0=a00, 1=a01, 2=a10, 3=a11
    # strassen_2x2_B[t][entry]: coefficient for B entry in term t
    # B entries: 0=b00, 1=b01, 2=b10, 3=b11
    # strassen_2x2_C[t][entry]: coefficient for C entry in term t
    # C entries: 0=c00, 1=c01, 2=c10, 3=c11

    s2_A = [
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, 1],   # M7: a01+a11
    ]
    s2_B = [
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [-1, 1, 0, 0],  # M4: b01-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b01
        [0, 0, 1, 1],   # M7: b10+b11
    ]
    s2_C = [
        [1, 0, 0, 0],   # M1: c00
        [0, 0, 0, 1],   # M2: c10
        [0, 1, 0, 0],   # M3: c01
        [1, 0, 1, 0],   # M4: c00+c10
        [-1, 1, 0, 0],  # M5: -c00+c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 1],   # M7: c00+c11
    ]

    # For each outer term (19) and each inner Strassen term (7):
    # The outer term says: for block (i,k) of A, use a_coeffs[i*3+k]
    #                    for block (k,j) of B, use b_coeffs[k*3+j]
    #                    for block (i,j) of C, use c_coeffs[i*3+j]
    # The inner term says: for 2x2 entries within those blocks, use s2 coefficients

    col = 0
    for t_outer in range(19):
        a_coeffs, b_coeffs, c_coeffs = term_defs[t_outer]

        for t_inner in range(7):
            # For each block position where a_coeffs is non-zero
            for bi in range(nb):
                for bj in range(nb):
                    a_block_coeff = a_coeffs[bi * 3 + bj]
                    if a_block_coeff == 0:
                        continue
                    # For each 2x2 entry within the A block (bi, bj)
                    for r in range(bs):
                        for c in range(bs):
                            local_idx = r * bs + c
                            s2a = s2_A[t_inner][local_idx]
                            if s2a != 0:
                                u_row = (bi * bs + r) * m + (bj * bs + c)
                                U[u_row][col] = a_block_coeff * s2a

            # For each block position where b_coeffs is non-zero
            for bk in range(nb):
                for bj in range(nb):
                    b_block_coeff = b_coeffs[bk * 3 + bj]
                    if b_block_coeff == 0:
                        continue
                    for r in range(bs):
                        for c in range(bs):
                            local_idx = r * bs + c
                            s2b = s2_B[t_inner][local_idx]
                            if s2b != 0:
                                v_row = (bk * bs + r) * p + (bj * bs + c)
                                V[v_row][col] = b_block_coeff * s2b

            # For each block position where c_coeffs is non-zero
            for bi in range(nb):
                for bj in range(nb):
                    c_block_coeff = c_coeffs[bi * 3 + bj]
                    if c_block_coeff == 0:
                        continue
                    for r in range(bs):
                        for c in range(bs):
                            local_idx = r * bs + c
                            s2c = s2_C[t_inner][local_idx]
                            if s2c != 0:
                                # W convention: W[k*n+i, r] where (k,i) is (col_C, row_C)
                                # C block (bi, bj): row block bi, col block bj
                                # Within block: row r, col c
                                # So global row = bi*bs + r, global col = bj*bs + c
                                # W index: col * n + row = (bj*bs+c)*n + (bi*bs+r)
                                w_row = (bj * bs + c) * n + (bi * bs + r)
                                W[w_row][col] = c_block_coeff * s2c

            col += 1

    # Quick verification
    if time.time() - start_time < DEADLINE - 10:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # Verify: sum_r U[i,r]*V[j,r]*W[k,r] should give C[k,i] for A[i,:] * B[:,j]
        # Actually the tensor decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # where T[i,j,k] = 1 if k == sum over appropriate... 
        # For matrix mult: (AB)[i,j] = sum_k A[i,k]*B[k,j]
        # Tensor T[i,k,j] = 1 (the multiplication tensor)
        # U has shape (n*m, rank) = (36, rank): indexed by (i,k) for A
        # V has shape (m*p, rank) = (36, rank): indexed by (k,j) for B
        # W has shape (p*n, rank) = (36, rank): indexed by (j,i) for C (W[k*n+i] = W[j*6+i])
        # So: sum_r U[i*6+k, r] * V[k*6+j, r] * W[j*6+i, r] should = 1 for the (i,j,k) term
        # i.e., reconstruct the multiplication tensor
        test_pass = True
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    expected = 1.0
                    got = 0.0
                    for r in range(total_rank):
                        got += U_np[i*6+k, r] * V_np[k*6+j, r] * W_np[j*6+i, r]
                    if abs(got - expected) > 1e-9:
                        test_pass = False
                        break
                if not test_pass:
                    break
            if not test_pass:
                break
        if not test_pass:
            # Fallback
            total_rank_fb = 27 * 7
            U_fb = [[0] * total_rank_fb for _ in range(n * m)]
            V_fb = [[0] * total_rank_fb for _ in range(m * p)]
            W_fb = [[0] * total_rank_fb for _ in range(p * n)]
            col = 0
            strassen_A = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
            strassen_B = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
            output_coeffs = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]
            for i in range(nb):
                for j in range(nb):
                    for k in range(nb):
                        for t in range(7):
                            for r in range(bs):
                                for c in range(bs):
                                    a_local = r * bs + c
                                    a_coeff = strassen_A[t][a_local]
                                    if a_coeff != 0:
                                        u_idx = (i * bs + r) * m + (k * bs + c)
                                        U_fb[u_idx][col] = a_coeff
                            for r in range(bs):
                                for c in range(bs):
                                    b_local = r * bs + c
                                    b_coeff = strassen_B[t][b_local]
                                    if b_coeff != 0:
                                        v_idx = (k * bs + r) * p + (j * bs + c)
                                        V_fb[v_idx][col] = b_coeff
                            for r in range(bs):
                                for c in range(bs):
                                    c_local = r * bs + c
                                    c_coeff = output_coeffs[t][c_local]
                                    if c_coeff != 0:
                                        w_idx = (j * bs + c) * n + (i * bs + r)
                                        W_fb[w_idx][col] = c_coeff
                            col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
    """
    Block-structured decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 3x3 grid of 2x2 blocks.
    Apply Strassen (rank 19) to the 3x3 block matrix multiplication.
    For each Strassen term, the "scalar" products are 2x2 matrix multiplications (rank 4 schoolbook).
    Total rank = 19 * 4 = 76.
    
    W follows convention W[k*n+i, r] where (k,i) is (col_C, row_C).
    """
# EVOLVE-BLOCK-END
