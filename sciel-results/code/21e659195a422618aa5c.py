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

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def build_2x2x1_schoolbook():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        
        # C00 = A00*B00 + A01*B10
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        
        # C10 = A10*B00 + A11*B10
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        Split B into B_left (2x1) and B_right (2x2).
        Use schoolbook (rank 4) for left, Strassen (rank 7) for right.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]  # 2*3
        W = [[0]*rank for _ in range(6)]  # 3*2
        
        # Part 1: 2x2 * 2x1 -> 2x1 (columns 0-3 of rank)
        U1, V1, W1 = build_2x2x1_schoolbook()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0:
                    U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0:
                    V[r*3][c] += V1[r][c]  # V row j*3+k, k=0 for first col
            for r in range(2):
                if W1[r][c] != 0:
                    W[r*2][c] += W1[r][c]  # W row k*n+i, n=2, k=0,1 -> rows 0,2? No.
                    # W shape is (p*n, rank) = (3*2, rank) = (6, rank)
                    # W[k*n + i, r] where k in 0..1, i in 0..1
                    # So W row index is k*2 + i
                    # W1 has shape (2, rank) where row is i (output row)
                    # We need to map W1[i] to W[k*2+i] for k=0? 
                    # Wait, C is 2x3. C[i,k]. i in 0..1, k in 0..2.
                    # W index is k*n + i = k*2 + i.
                    # For the first column (k=0), W index is 0*2 + i = i.
                    # So W1[i] maps to W[i].
                    pass
        
        # Let's redo W mapping carefully.
        # W1 is (2, 4) for 2x1 output. Rows are i=0,1.
        # For the full 2x3 output, W is (6, 11).
        # The first column of C (k=0) corresponds to W rows 0*2+0=0 and 0*2+1=1.
        # So W1[0] -> W[0], W1[1] -> W[1].
        for c in range(4):
            for r in range(2):
                if W1[r][c] != 0:
                    W[r][c] += W1[r][c]

        # Part 2: 2x2 * 2x2 -> 2x2 (columns 4-10 of rank)
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0:
                    U[r][base + c] += U2[r][c]
            # V2 is (4, 7) for 2x2 input B. Rows are j*2+k, j=0,1, k=0,1.
            # V for 2x3 input is (6, 11). Rows are j*3+k, j=0,1, k=0,1,2.
            # We are multiplying A (2x2) by B_right (2x2) which is columns 1,2 of B (2x3).
            # So B_right[j, k_local] corresponds to B[j, k_local + 1].
            # V2 row index is j*2 + k_local.
            # V row index is j*3 + (k_local + 1).
            for r in range(4):
                if V2[r][c] != 0:
                    j = r // 2
                    k_local = r % 2
                    v_idx = j * 3 + (k_local + 1)
                    V[v_idx][base + c] += V2[r][c]
            # W2 is (4, 7) for 2x2 output C. Rows are k*2+i, k=0,1, i=0,1.
            # W for 2x3 output is (6, 11). Rows are k*2+i, k=0,1,2, i=0,1.
            # The 2x2 output corresponds to columns 1,2 of the 2x3 output.
            # So C_right[k_local, i] corresponds to C[k_local + 1, i].
            # W2 row index is k_local*2 + i.
            # W row index is (k_local + 1)*2 + i.
            for r in range(4):
                if W2[r][c] != 0:
                    k_local = r // 2
                    i = r % 2
                    w_idx = (k_local + 1) * 2 + i
                    W[w_idx][base + c] += W2[r][c]
        
        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6.
        Uses 7 Strassen steps for the 4x4 * 4x4 core, and handles the 4x2 strip.
        Actually, let's split 4x4 * 4x6 into:
        A (4x4) * B (4x6) -> C (4x6)
        Split B into B1 (4x4) and B2 (4x2).
        C1 = A * B1 (4x4)
        C2 = A * B2 (4x2)
        
        For C1, use 4x4 * 4x4 Strassen (rank 49).
        For C2, use 4x4 * 4x2.
        Split B2 into B2a (4x1) and B2b (4x1).
        C2a = A * B2a (4x1)
        C2b = A * B2b (4x1)
        
        4x4 * 4x1 can be done with 4x4 * 4x4 Strassen? No, that's overkill.
        4x4 * 4x1 schoolbook is rank 16.
        So total rank = 49 + 16 + 16 = 81.
        
        Can we do better?
        4x4 * 4x2.
        Split A into 2x2 blocks.
        A = [[A00, A01], [A10, A11]]
        B2 = [[B00, B01], [B10, B11]] where B00 is 2x2, etc.
        C2 = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
              [A10*B00 + A11*B10, A10*B01 + A11*B11]]
        
        This is 8 multiplications of 2x2 * 2x2.
        Each 2x2 * 2x2 can be done with Strassen (rank 7).
        Total rank for C2 = 8 * 7 = 56.
        Total rank = 49 + 56 = 105. Worse.
        
        Let's stick with the simple composition:
        4x4 * 4x4 (rank 49) + 4x4 * 4x2 (rank 32 schoolbook) = 81.
        
        Or use the 2x2x3 block (rank 11) and compose.
        4x4 * 4x6.
        Split A into 2x2 blocks (4 blocks).
        Split B into 2x2 blocks (12 blocks).
        This is complex.
        
        Let's use the 2x2x3 block (rank 11) to build 4x4x6.
        A (4x4) = [[A00, A01], [A10, A11]]
        B (4x6) = [[B00, B01, B02], [B10, B11, B12]] where B00 is 2x2, B01 is 2x2, B02 is 2x2.
        C (4x6) = [[A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12],
                   [A10*B00 + A11*B10, A10*B01 + A11*B11, A10*B02 + A11*B12]]
        
        Each term is a sum of two 2x2 * 2x2 products.
        We can use Strassen for each 2x2 * 2x2.
        But we have 8 such products.
        Total rank = 8 * 7 = 56.
        
        This is better than 81.
        
        Let's implement this.
        """
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x2_strassen()
        rank_sub = len(U_sub[0])
        total_rank = 8 * rank_sub
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # Terms: (A_block, B_block, C_block)
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 02, 10, 11, 12
        # C blocks: 00, 01, 02, 10, 11, 12
        terms = [
            (0, 0, 0), # A00 * B00 -> C00
            (0, 1, 1), # A00 * B01 -> C01
            (0, 2, 2), # A00 * B02 -> C02
            (1, 0, 0), # A01 * B10 -> C00
            (1, 1, 1), # A01 * B11 -> C01
            (1, 2, 2), # A01 * B12 -> C02
            (2, 0, 10), # A10 * B00 -> C10
            (2, 1, 11), # A10 * B01 -> C11
            (2, 2, 12), # A10 * B02 -> C12
            (3, 0, 10), # A11 * B10 -> C10
            (3, 1, 11), # A11 * B11 -> C11
            (3, 2, 12), # A11 * B12 -> C12
        ]
        
        # Wait, C has 6 columns. 2x6.
        # C blocks are 2x2.
        # C00, C01, C02 are top row blocks.
        # C10, C11, C12 are bottom row blocks.
        # So C block indices:
        # 0: C00 (rows 0-1, cols 0-1)
        # 1: C01 (rows 0-1, cols 2-3)
        # 2: C02 (rows 0-1, cols 4-5)
        # 3: C10 (rows 2-3, cols 0-1)
        # 4: C11 (rows 2-3, cols 2-3)
        # 5: C12 (rows 2-3, cols 4-5)
        
        # A blocks:
        # 0: A00 (rows 0-1, cols 0-1)
        # 1: A01 (rows 0-1, cols 2-3)
        # 2: A10 (rows 2-3, cols 0-1)
        # 3: A11 (rows 2-3, cols 2-3)
        
        # B blocks:
        # 0: B00 (rows 0-1, cols 0-1)
        # 1: B01 (rows 0-1, cols 2-3)
        # 2: B02 (rows 0-1, cols 4-5)
        # 3: B10 (rows 2-3, cols 0-1)
        # 4: B11 (rows 2-3, cols 2-3)
        # 5: B12 (rows 2-3, cols 4-5)
        
        terms = [
            (0, 0, 0), # A00 * B00 -> C00
            (0, 1, 1), # A00 * B01 -> C01
            (0, 2, 2), # A00 * B02 -> C02
            (1, 3, 0), # A01 * B10 -> C00
            (1, 4, 1), # A01 * B11 -> C01
            (1, 5, 2), # A01 * B12 -> C02
            (2, 0, 3), # A10 * B00 -> C10
            (2, 1, 4), # A10 * B01 -> C11
            (2, 2, 5), # A10 * B02 -> C12
            (3, 3, 3), # A11 * B10 -> C10
            (3, 4, 4), # A11 * B11 -> C11
            (3, 5, 5), # A11 * B12 -> C12
        ]
        
        # 12 terms, each rank 7. Total rank 84.
        total_rank = 12 * rank_sub
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        v_offset = {0: (0, 0), 1: (0, 2), 2: (0, 4), 3: (2, 0), 4: (2, 2), 5: (2, 4)}
        w_offset = {0: (0, 0), 1: (0, 2), 2: (0, 4), 3: (2, 0), 4: (2, 2), 5: (2, 4)}
        
        for t_idx, (a_block, b_block, c_block) in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    ro, co = u_offset[a_block]
                    g_u = (i_local + ro) * m + (j_local + co)
                    U[g_u][base_col + c] += u_val
                
                for v_local in range(4):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 2
                    k_local = v_local % 2
                    ro, co = v_offset[b_block]
                    g_v = (j_local + ro) * p + (k_local + co)
                    V[g_v][base_col + c] += v_val
                
                for w_local in range(4):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    ro, co = w_offset[c_block]
                    g_w = (k_local + ro) * n + (i_local + co)
                    W[g_w][base_col + c] += w_val
        
        return U, V, W

    U, V, W = build_4x4x6()

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
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
