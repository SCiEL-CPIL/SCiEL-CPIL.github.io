"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def time_remaining():
        return deadline - time.time()

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def build_4x4x4_strassen():
        """Rank-49 Strassen decomposition for 4x4 * 4x4 -> 4x4."""
        U_sub, V_sub, W_sub = build_2x2x2_strassen()
        rank_sub = 7
        total_rank = 7 * rank_sub  # 49

        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(16)]
        W = [[0]*total_rank for _ in range(16)]

        # Strassen 4x4x4: 7 recursive calls on 2x2x2
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]], C = A*B
        # M1 = A11*(B11+B22), M2 = (A21+A22)*B11, M3 = (A11+A12)*B22
        # M4 = A22*(B11-B22), M5 = (A11+A12)*(B22-B11)
        # M6 = (A21-A11)*B12, M7 = (A12-A22)*(B21+B22)
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M6, C22 = M1-M2+M3+M4

        # Each term: which 2x2 sub-blocks of A, B, C are involved and with what coefficients
        # A sub-blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B sub-blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # C sub-blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        terms = [
            # M1 = (A11+A22)*(B11+B22)
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            # M2 = (A21+A22)*B11
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1)]},
            # M3 = A11*(B12-B22)
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, -1)]},
            # M4 = A22*(B21-B11)
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, -1), (3, 1)]},
            # M5 = (A11+A12)*B22
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, 1), (1, 1)]},
            # M6 = (A21-A11)*(B11+B12)
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(2, 1)]},
            # M7 = (A12-A22)*(B21+B22)
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, -1), (3, 1)]},
        ]

        # Offset mapping: sub-block index -> (row_offset, col_offset) in 4x4
        a_off = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        b_off = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        c_off = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = a_off[block]
                        g_u = (i_local + ro) * 4 + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff

                for v_local in range(4):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 2
                    k_local = v_local % 2
                    for block, coeff in term['B']:
                        ro, co = b_off[block]
                        g_v = (j_local + ro) * 4 + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff

                for w_local in range(4):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = c_off[block]
                        g_w = (k_local + ro) * 4 + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W

    def build_4x4x2():
        """Rank-28 decomposition for 4x4 * 4x2 -> 4x2.
        Split B (4x2) into two 2x2 blocks B1 (top) and B2 (bottom).
        C = A*B = [A*B1; ... ] wait, B is 4x2, so B = [B_top; B_bot] where each is 2x2.
        C = A * B, A is 4x4, B is 4x2, C is 4x2.
        Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
        B = [B1; B2] where B1, B2 are 2x2
        C = [C1; C2] where C1, C2 are 2x2
        C1 = A11*B1 + A12*B2
        C2 = A21*B1 + A22*B2
        Use Strassen for each 2x2*2x2 product. 4 products * rank 7 = 28.
        """
        U_sub, V_sub, W_sub = build_2x2x2_strassen()
        rank_sub = 7
        total_rank = 4 * rank_sub  # 28

        U = [[0]*total_rank for _ in range(16)]  # 4*4
        V = [[0]*total_rank for _ in range(8)]   # 4*2
        W = [[0]*total_rank for _ in range(8)]   # 2*4 (p*n = 2*4)

        # 4 products: (A_block, B_block, C_block)
        # C1 = A11*B1 + A12*B2: two Strassen products
        # C2 = A21*B1 + A22*B2: two Strassen products
        products = [
            # (A_sub, B_sub, C_sub) - sub indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            (0, 0, 0),  # A11 * B1 -> C1
            (1, 1, 0),  # A12 * B2 -> C1
            (2, 0, 1),  # A21 * B1 -> C2
            (3, 1, 1),  # A22 * B2 -> C2
        ]

        a_off = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        # B is 4x2: B_sub 0 = rows 0-1, cols 0-1; B_sub 1 = rows 2-3, cols 0-1
        b_off = {0: (0, 0), 1: (2, 0)}
        # C is 4x2, W indexed by [k*n+i] where k in 0..1 (p=2), i in 0..3 (n=4)
        # C1 = top 2 rows of C (i=0,1), C2 = bottom 2 rows (i=2,3)
        # W row = k*4 + i
        # For C1: i = i_local, so offset on i is 0
        # For C2: i = i_local + 2, so offset on i is 2
        c_off = {0: (0, 0), 1: (0, 2)}

        for prod_idx, (a_sub, b_sub, c_sub) in enumerate(products):
            base_col = prod_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    ro, co = a_off[a_sub]
                    g_u = (i_local + ro) * 4 + (j_local + co)
                    U[g_u][base_col + c] += u_val

                for v_local in range(4):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 2
                    k_local = v_local % 2
                    ro, co = b_off[b_sub]
                    g_v = (j_local + ro) * 2 + (k_local + co)
                    V[g_v][base_col + c] += v_val

                for w_local in range(4):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    ro, co = c_off[c_sub]
                    g_w = (k_local + ro) * 4 + (i_local + co)
                    W[g_w][base_col + c] += w_val

        return U, V, W

    def build_4x4x6():
        """Rank-77 decomposition: 4x4x4 (rank 49) + 4x4x2 (rank 28)."""
        U44, V44, W44 = build_4x4x4_strassen()
        U42, V42, W42 = build_4x4x2()

        rank1 = len(U44[0])  # 49
        rank2 = len(U42[0])  # 28
        total_rank = rank1 + rank2  # 77

        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(24)]  # 4*6
        W = [[0]*total_rank for _ in range(24)]  # 6*4

        # Copy 4x4x4 part (columns 0..48)
        for r in range(16):
            for c in range(rank1):
                U[r][c] = U44[r][c]
        for r in range(16):
            for c in range(rank1):
                V[r][c] = V44[r][c]
        for r in range(16):
            for c in range(rank1):
                W[r][c] = W44[r][c]

        # Copy 4x4x2 part (columns 49..76)
        # V for 4x4x2 has shape (8, 28) - rows are j*p+k where j in 0..3, k in 0..1
        # We need to place these into V[24] where k ranges 0..5
        # The 4x4x2 handles columns 0,1 of B. Place into V rows with k=0,1
        base_col = rank1
        for r in range(16):
            for c in range(rank2):
                U[r][base_col + c] = U42[r][c]
        # V42 has 8 rows: j*2+k for j in 0..3, k in 0..1
        # Map to V[j*6+k]
        for r in range(8):
            j = r // 2
            k = r % 2
            for c in range(rank2):
                V[j*6 + k][base_col + c] = V42[r][c]
        # W42 has 8 rows: k*4+i for k in 0..1, i in 0..3
        # Map to W[k*4+i] directly (same indexing)
        for r in range(8):
            for c in range(rank2):
                W[r][base_col + c] = W42[r][c]

        return U, V, W

    U, V, W = build_4x4x6()

    # Verify the decomposition using numpy for speed
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_arr[i*m+j, :]
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    prod = u_row * v_row * w_row
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
