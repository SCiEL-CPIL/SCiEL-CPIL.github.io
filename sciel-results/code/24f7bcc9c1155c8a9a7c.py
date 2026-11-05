import time

def construct():
    deadline = time.time() + 118.0

    n, m, p = 3, 4, 4

    # Schoolbook solution as fallback (Rank 48)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication (Strassen)."""
        terms = []
        # p1 = (a11+a22)*(b11+b22)
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # p2 = (a21+a22)*b11
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # p3 = a11*(b12-b22)
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # p4 = a22*(b21-b11)
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # p5 = (a11+a12)*b22
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # p6 = (a21-a11)*(b11+b12)
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # p7 = (a12-a22)*(b21+b22)
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    # Rank 44 Decomposition
    # C[0:2, :] computed by Strassen on 2x4 * 4x4 blocks
    # C[2, :] computed by Schoolbook on 1x4 * 4x4
    
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 44

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        """Add a rank-7 2x2*2x2 block.
        a_row_offset: row offset in A (0 or 2)
        b_col_offset: col offset in B (0 or 2)
        j_offset: row offset in B / col offset in A (0 or 2)
        c_row_offset: row offset in C (0 or 2)
        c_col_offset: col offset in C (0 or 2)
        """
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            col = col_start + r
            for (ai, aj), val in u_dict.items():
                a_row = a_row_offset + ai
                if a_row < n:
                    U[a_row * m + (j_offset + aj)][col] += val
            for (bj, bk), val in v_dict.items():
                V[(j_offset + bj) * p + (b_col_offset + bk)][col] += val
            for (ci, ck), val in w_dict.items():
                c_row = c_row_offset + ci
                if c_row < n:
                    W[(c_col_offset + ck) * n + c_row][col] += val
        return col_start + 7

    col = 0
    # C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 0:2]
    col = add_rank7_block(0, 0, 0, 0, 0, col)
    # A[0:2, 2:4] * B[2:4, 0:2]
    col = add_rank7_block(0, 0, 2, 0, 0, col)
    
    # C[0:2, 2:4]
    # A[0:2, 0:2] * B[0:2, 2:4]
    col = add_rank7_block(0, 2, 0, 0, 2, col)
    # A[0:2, 2:4] * B[2:4, 2:4]
    col = add_rank7_block(0, 2, 2, 0, 2, col)
    
    # C[2, 0:2]
    # A[2, 0:2] * B[0:2, 0:2] (Schoolbook 1x2 * 2x2 -> 1x2)
    # Rank 4
    for j in range(2):
        for k in range(2):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # C[2, 2:4]
    # A[2, 2:4] * B[2:4, 2:4] (Schoolbook 1x2 * 2x2 -> 1x2)
    # Rank 4
    for j in range(2, 4):
        for k in range(2, 4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # Wait, the above schoolbook blocks for C[2, :] only cover B[0:2, 0:2] and B[2:4, 2:4].
    # I missed B[0:2, 2:4] and B[2:4, 0:2] for the row 2 of C.
    # C[2, 0:2] = A[2, 0:2] B[0:2, 0:2] + A[2, 2:4] B[2:4, 0:2]
    # C[2, 2:4] = A[2, 0:2] B[0:2, 2:4] + A[2, 2:4] B[2:4, 2:4]
    
    # My loop above:
    # 1. j in 0..1, k in 0..1. (A[2,0:2], B[0:2,0:2]) -> C[2,0:2]. Correct.
    # 2. j in 2..3, k in 2..3. (A[2,2:4], B[2:4,2:4]) -> C[2,2:4]. Correct.
    
    # I am missing:
    # A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    
    # Let's add these.
    # A[2, 2:4] * B[2:4, 0:2]
    for j in range(2, 4):
        for k in range(0, 2):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # A[2, 0:2] * B[0:2, 2:4]
    for j in range(0, 2):
        for k in range(2, 4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # Total rank: 28 (Strassen) + 4 + 4 + 4 + 4 = 44.
    
    return {"U": U, "V": V, "W": W}
