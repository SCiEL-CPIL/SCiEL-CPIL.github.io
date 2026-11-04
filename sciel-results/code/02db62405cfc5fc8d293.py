"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1.
        Returns (U, V, W) as lists of columns, each a list of length (rows).
        U: 2*2=4 rows, V: 2*1=2 rows, W: 1*2=2 rows.
        Indexing: U[i*2+j], V[j*1+k]=V[j], W[k*2+i].
        """
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]

        # Term 0: A[0,0]*B[0] -> C[0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 1: A[0,1]*B[1] -> C[0]
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        # Term 2: A[1,0]*B[0] -> C[1]
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        # Term 3: A[1,1]*B[1] -> C[1]
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1

        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
        U: 2*2=4 rows, V: 2*2=4 rows, W: 2*2=4 rows.
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i].
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]

        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        Split B into B1 (2x1) and B2 (2x2).
        A*B1 uses rank-4 schoolbook.
        A*B2 uses rank-7 Strassen.
        U: 2*2=4 rows, V: 2*3=6 rows, W: 3*2=6 rows.
        Indexing: U[i*2+j], V[j*3+k], W[k*2+i].
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        # Part 1: A*B1 (rank 4), B1 is column 0 of B, C1 is column 0 of C.
        # B1 row j -> V index j*3+0
        # C1 row i -> W index 0*2+i = i
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0:
                    U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0:
                    V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0:
                    W[r][c] += W1[r][c]

        # Part 2: A*B2 (rank 7), B2 is columns 1,2 of B, C2 is columns 1,2 of C.
        # B2[j,k] -> B[j, k+1] -> V index j*3+(k+1)
        # C2[i,k] -> C[i, k+1] -> W index (k+1)*2+i
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        # V2 index j*2+k maps to V index j*3+(k+1)
        v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
        # W2 index k*2+i maps to W index (k+1)*2+i
        w2_map = {0: 2, 1: 3, 2: 4, 3: 5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0:
                    U[r][base + c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0:
                    V[v2_map[r]][base + c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0:
                    W[w2_map[r]][base + c] += W2[r][c]

        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6.
        Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x3, C into 2x2 blocks of 2x3.
        7 Strassen sub-products, each 2x2 * 2x3 -> 2x3 with rank 11.
        Total rank = 7 * 11 = 77.
        U: 4*4=16 rows, V: 4*6=24 rows, W: 6*4=24 rows.
        Indexing: U[i*4+j], V[j*6+k], W[k*4+i].
        """
        n, m, p = 4, 4, 6
        n2, m2, p2 = 2, 2, 3

        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])  # 11
        total_rank = 7 * rank_sub  # 77

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        # Strassen terms for 4x4 * 4x6:
        # A blocks: A11 (rows 0:2, cols 0:2), A12 (rows 0:2, cols 2:4)
        #           A21 (rows 2:4, cols 0:2), A22 (rows 2:4, cols 2:4)
        # B blocks: B11 (rows 0:2, cols 0:3), B12 (rows 0:2, cols 3:6)
        #           B21 (rows 2:4, cols 0:3), B22 (rows 2:4, cols 3:6)
        # C blocks: C11 (rows 0:2, cols 0:3), C12 (rows 0:2, cols 3:6)
        #           C21 (rows 2:4, cols 0:3), C22 (rows 2:4, cols 3:6)
        #
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # For each Strassen term, we have coefficients on A-blocks, B-blocks, C-blocks.
        # Each block is a 2x2 (for A) or 2x3 (for B, C) sub-matrix.
        # The sub-decomposition U_sub/V_sub/W_sub operates on local indices:
        #   U_sub: i2*2+j2 where i2 in 0..1, j2 in 0..1
        #   V_sub: j2*3+k2 where j2 in 0..1, k2 in 0..2
        #   W_sub: k2*2+i2 where k2 in 0..2, i2 in 0..1

        # Strassen term definitions: (u_blocks, v_blocks, w_blocks)
        # Each is a dict mapping block index to coefficient.
        # A block indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B block indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C block indices: 0=C11, 1=C12, 2=C21, 3=C22

        strassen_terms = [
            # M1 = (A11+A22)(B11+B22)
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            # M2 = (A21+A22)B11
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            # M3 = A11(B12-B22)
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            # M4 = A22(B21-B11)
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            # M5 = (A11+A12)B22
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            # M6 = (A21-A11)(B11+B12)
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            # M7 = (A12-A22)(B21+B22)
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        # Block offsets:
        # A block (br, bc) -> row offset br*2, col offset bc*2
        # B block (br, bc) -> row offset br*2, col offset bc*3
        # C block (br, bc) -> row offset br*2, col offset bc*3

        # Block index to (br, bc):
        a_block_pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        b_block_pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        c_block_pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}

        for term_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_terms):
            base_rank = term_idx * rank_sub
            for c in range(rank_sub):
                u_col = [0] * (n*m)
                v_col = [0] * (m*p)
                w_col = [0] * (p*n)

                # Fill U from u_blocks
                for blk_idx, coeff_u in u_blocks.items():
                    br, bc = a_block_pos[blk_idx]
                    for i2 in range(2):
                        for j2 in range(2):
                            val = U_sub[i2*2+j2][c]
                            if val != 0:
                                i_global = br*2 + i2
                                j_global = bc*2 + j2
                                u_col[i_global*m + j_global] += coeff_u * val

                # Fill V from v_blocks
                for blk_idx, coeff_v in v_blocks.items():
                    br, bc = b_block_pos[blk_idx]
                    for j2 in range(2):
                        for k2 in range(3):
                            val = V_sub[j2*3+k2][c]
                            if val != 0:
                                j_global = br*2 + j2
                                k_global = bc*3 + k2
                                v_col[j_global*p + k_global] += coeff_v * val

                # Fill W from w_blocks
                for blk_idx, coeff_w in w_blocks.items():
                    br, bc = c_block_pos[blk_idx]
                    for k2 in range(3):
                        for i2 in range(2):
                            val = W_sub[k2*2+i2][c]
                            if val != 0:
                                i_global = br*2 + i2
                                k_global = bc*3 + k2
                                w_col[k_global*n + i_global] += coeff_w * val

                # Store as columns
                for i in range(n*m):
                    U[i][base_rank + c] = u_col[i]
                for j in range(m*p):
                    V[j][base_rank + c] = v_col[j]
                for k in range(p*n):
                    W[k][base_rank + c] = w_col[k]

        return U, V, W

    if _time_remaining() > 5.0:
        U, V, W = build_4x4x6()
    else:
        # Fallback: return empty (should not happen)
        U = [[0]*1 for _ in range(n*m)]
        V = [[0]*1 for _ in range(m*p)]
        W = [[0]*1 for _ in range(p*n)]

    return {"U": U, "V": V, "W": W}
