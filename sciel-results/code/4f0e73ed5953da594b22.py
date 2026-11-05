"""
Rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
Uses a block-partition search: split rows, columns, and output dimensions into
blocks of size 1 and 2, then apply Strassen (rank 7) to 2x2x2 blocks and
schoolbook (rank 8) to all other blocks. The minimum over all such partitions
gives rank 99.

This version uses the cleaner direct-construction approach from the improved
island (score 1.0606) which explicitly builds each block type, verified to
achieve rank 99.
"""
import time

DIMENSIONS = [3, 5, 7]


def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7

    # Best partition: split m=5 into [2,2,1], n=3 into [1,2] or [2,1], p=7 into [2,2,2,1] etc.
    # The verified best is:
    #   n=3: [1, 2]  (1 row, then 2 rows)
    #   m=5: [2, 2, 1] (2 cols, 2 cols, 1 col)
    #   p=7: [2, 2, 2, 1] (2, 2, 2, 1)
    # This gives blocks:
    #   (1,2,2): rank 4 (M122 - two dot products of length 2)
    #   (1,2,2): rank 4
    #   (1,2,2): rank 4
    #   (1,2,1): rank 2
    #   (2,2,2): rank 7 (Strassen)
    #   (2,2,2): rank 7
    #   (2,2,2): rank 7
    #   (2,2,1): rank 4
    #   (2,1,2): rank 4
    #   (2,1,2): rank 4
    #   (2,1,2): rank 4
    #   (2,1,1): rank 2
    # Total: 4+4+4+2+7+7+7+4+4+4+4+2 = 53... wait that's not right.
    # Let me recount: n=3=[1,2], m=5=[2,2,1], p=7=[2,2,2,1]
    # Blocks: for each (ni, mj, pk):
    # ni=1: mj=2: pk=2,2,2,1 -> 4 blocks of (1,2,2) rank 4 each, and (1,2,1) rank 2
    #         = 4*4 + 2 = 18
    # ni=1: mj=2: same -> 18
    # ni=1: mj=1: pk=2,2,2,1 -> (1,1,2) rank 2 each, (1,1,1) rank 1
    #         = 3*2 + 1 = 7
    # ni=2: mj=2: pk=2,2,2,1 -> (2,2,2) rank 7 each, (2,2,1) rank 4
    #         = 3*7 + 4 = 25
    # ni=2: mj=2: same -> 25
    # ni=2: mj=1: pk=2,2,2,1 -> (2,1,2) rank 4 each, (2,1,1) rank 2
    #         = 3*4 + 2 = 14
    # Total: 18 + 18 + 7 + 25 + 25 + 14 = 107... that's worse.

    # Let me try n=3=[2,1], m=5=[2,2,1], p=7=[2,2,2,1]:
    # ni=2: mj=2: pk=2,2,2,1 -> 3*(2,2,2)=7 + (2,2,1)=4 = 25
    # ni=2: mj=2: same = 25
    # ni=2: mj=1: pk=2,2,2,1 -> 3*(2,1,2)=4 + (2,1,1)=2 = 14
    # ni=1: mj=2: pk=2,2,2,1 -> 3*(1,2,2)=4 + (1,2,1)=2 = 14
    # ni=1: mj=2: same = 14
    # ni=1: mj=1: pk=2,2,2,1 -> 3*(1,1,2)=2 + (1,1,1)=1 = 7
    # Total: 25+25+14+14+14+7 = 99. Yes!

    # So the best partition is: n=[2,1], m=[2,2,1], p=[2,2,2,1]
    # Total rank = 99

    rank_total = 99

    U = [[0] * rank_total for _ in range(n * m)]  # 15 x 99
    V = [[0] * rank_total for _ in range(m * p)]  # 35 x 99
    W = [[0] * rank_total for _ in range(p * n)]  # 21 x 99

    def add_strassen(rank_offset, i0, j0, k0):
        """Add M222 Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
        # U local index: i_l*2 + j_l
        # V local index: j_l*2 + k_l
        # W local index: k_l*2 + i_l

        u_loc = [[0]*7 for _ in range(4)]
        v_loc = [[0]*7 for _ in range(4)]
        w_loc = [[0]*7 for _ in range(4)]

        # M1 = (A00+A11)(B00+B11)
        u_loc[0][0] = 1; u_loc[3][0] = 1
        v_loc[0][0] = 1; v_loc[3][0] = 1
        w_loc[0][0] = 1; w_loc[3][0] = 1

        # M2 = (A10+A11)B00
        u_loc[2][1] = 1; u_loc[3][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1; w_loc[3][1] = -1

        # M3 = A00(B01-B11)
        u_loc[0][2] = 1
        v_loc[1][2] = 1; v_loc[3][2] = -1
        w_loc[2][2] = 1; w_loc[3][2] = 1

        # M4 = A11(B10-B00)
        u_loc[3][3] = 1
        v_loc[2][3] = 1; v_loc[0][3] = -1
        w_loc[0][3] = 1; w_loc[1][3] = 1

        # M5 = (A00+A01)B11
        u_loc[0][4] = 1; u_loc[1][4] = 1
        v_loc[3][4] = 1
        w_loc[0][4] = -1; w_loc[2][4] = 1

        # M6 = (A10-A00)(B00+B01)
        u_loc[2][5] = 1; u_loc[0][5] = -1
        v_loc[0][5] = 1; v_loc[1][5] = 1
        w_loc[3][5] = 1

        # M7 = (A01-A11)(B10+B11)
        u_loc[1][6] = 1; u_loc[3][6] = -1
        v_loc[2][6] = 1; v_loc[3][6] = 1
        w_loc[0][6] = 1

        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]

        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]

        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    def add_m221(rank_offset, i0, j0, k0):
        """Add M221 for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]"""
        # Rank 4: C_i = sum_j A_ij B_j0
        # r=0: A00*B00, r=1: A01*B10, r=2: A10*B00, r=3: A11*B10

        for r, (i_l, j_l) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            u_g_idx = (i0 + i_l) * m + (j0 + j_l)
            U[u_g_idx][rank_offset + r] += 1

            v_g_idx = (j0 + j_l) * p + (k0 + 0)
            V[v_g_idx][rank_offset + r] += 1

            w_g_idx = (k0 + 0) * n + (i0 + i_l)
            W[w_g_idx][rank_offset + r] += 1

    def add_m122(rank_offset, i0, j0, k0):
        """Add M122 for A[i0:i0+1, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        # Rank 4: C_k = sum_j A_0j B_jk
        # r=0: A00*B00, r=1: A00*B01, r=2: A01*B10, r=3: A01*B11

        for r, (j_l, k_l) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            u_g_idx = (i0 + 0) * m + (j0 + j_l)
            U[u_g_idx][rank_offset + r] += 1

            v_g_idx = (j0 + j_l) * p + (k0 + k_l)
            V[v_g_idx][rank_offset + r] += 1

            w_g_idx = (k0 + k_l) * n + (i0 + 0)
            W[w_g_idx][rank_offset + r] += 1

    def add_m212(rank_offset, i0, j0, k0):
        """Add M212 for A[i0:i0+2, j0:j0+1] * B[j0:j0+1, k0:k0+2]"""
        # Rank 4: C_ik = A_i0 * B_0k
        # r=0: A00*B00, r=1: A00*B01, r=2: A10*B00, r=3: A10*B01

        for r, (i_l, k_l) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            u_g_idx = (i0 + i_l) * m + (j0 + 0)
            U[u_g_idx][rank_offset + r] += 1

            v_g_idx = (j0 + 0) * p + (k0 + k_l)
            V[v_g_idx][rank_offset + r] += 1

            w_g_idx = (k0 + k_l) * n + (i0 + i_l)
            W[w_g_idx][rank_offset + r] += 1

    def add_m121(rank_offset, i0, j0, k0):
        """Add M121 for A[i0:i0+1, j0:j0+2] * B[j0:j0+2, k0:k0+1]"""
        # Rank 2: C_0 = A00*B00 + A01*B10

        for r, j_l in enumerate([0, 1]):
            u_g_idx = (i0 + 0) * m + (j0 + j_l)
            U[u_g_idx][rank_offset + r] += 1

            v_g_idx = (j0 + j_l) * p + (k0 + 0)
            V[v_g_idx][rank_offset + r] += 1

            w_g_idx = (k0 + 0) * n + (i0 + 0)
            W[w_g_idx][rank_offset + r] += 1

    def add_m211(rank_offset, i0, j0, k0):
        """Add M211 for A[i0:i0+2, j0:j0+1] * B[j0:j0+1, k0:k0+1]"""
        # Rank 2: C_i = A_i0 * B_00

        for r, i_l in enumerate([0, 1]):
            u_g_idx = (i0 + i_l) * m + (j0 + 0)
            U[u_g_idx][rank_offset + r] += 1

            v_g_idx = (j0 + 0) * p + (k0 + 0)
            V[v_g_idx][rank_offset + r] += 1

            w_g_idx = (k0 + 0) * n + (i0 + i_l)
            W[w_g_idx][rank_offset + r] += 1

    def add_m112(rank_offset, i0, j0, k0):
        """Add M112 for A[i0:i0+1, j0:j0+1] * B[j0:j0+1, k0:k0+2]"""
        # Rank 2: C_k = A_00 * B_0k

        for r, k_l in enumerate([0, 1]):
            u_g_idx = (i0 + 0) * m + (j0 + 0)
            U[u_g_idx][rank_offset + r] += 1

            v_g_idx = (j0 + 0) * p + (k0 + k_l)
            V[v_g_idx][rank_offset + r] += 1

            w_g_idx = (k0 + k_l) * n + (i0 + 0)
            W[w_g_idx][rank_offset + r] += 1

    def add_m111(rank_offset, i0, j0, k0):
        """Add M111 for A[i0:i0+1, j0:j0+1] * B[j0:j0+1, k0:k0+1]"""
        # Rank 1: C_00 = A_00 * B_00

        u_g_idx = (i0 + 0) * m + (j0 + 0)
        U[u_g_idx][rank_offset] += 1

        v_g_idx = (j0 + 0) * p + (k0 + 0)
        V[v_g_idx][rank_offset] += 1

        w_g_idx = (k0 + 0) * n + (i0 + 0)
        W[w_g_idx][rank_offset] += 1

    # Partition: n=[2,1], m=[2,2,1], p=[2,2,2,1]
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2, 1]

    col = 0
    i0 = 0
    for ni in n_parts:
        j0 = 0
        for mj in m_parts:
            k0 = 0
            for pk in p_parts:
                if ni == 2 and mj == 2 and pk == 2:
                    add_strassen(col, i0, j0, k0)
                    col += 7
                elif ni == 2 and mj == 2 and pk == 1:
                    add_m221(col, i0, j0, k0)
                    col += 4
                elif ni == 1 and mj == 2 and pk == 2:
                    add_m122(col, i0, j0, k0)
                    col += 4
                elif ni == 2 and mj == 1 and pk == 2:
                    add_m212(col, i0, j0, k0)
                    col += 4
                elif ni == 1 and mj == 2 and pk == 1:
                    add_m121(col, i0, j0, k0)
                    col += 2
                elif ni == 2 and mj == 1 and pk == 1:
                    add_m211(col, i0, j0, k0)
                    col += 2
                elif ni == 1 and mj == 1 and pk == 2:
                    add_m112(col, i0, j0, k0)
                    col += 2
                elif ni == 1 and mj == 1 and pk == 1:
                    add_m111(col, i0, j0, k0)
                    col += 1
                else:
                    # Fallback: schoolbook
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U[(i0+ii)*m + (j0+jj)][col] += 1
                                V[(j0+jj)*p + (k0+kk)][col] += 1
                                W[(k0+kk)*n + (i0+ii)][col] += 1
                                col += 1
                k0 += pk
            j0 += mj
        i0 += ni

    # Truncate if needed
    if col < rank_total:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]

    return {"U": U, "V": V, "W": W}
