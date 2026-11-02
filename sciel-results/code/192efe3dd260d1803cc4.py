"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS  # 4, 5, 6

    # Strategy:
    # Split 4x5x6 into three disjoint regions:
    # - Region A: i in [0,4), j in [0,4), k in [0,4)  -> 4x4x4 (Strassen rank 49)
    # - Region B: i in [0,4), j in [4,5), k in [0,6)  -> 4x1x6 (schoolbook rank 24)
    # - Region C: i in [0,4), j in [0,4), k in [4,6)  -> 4x4x2 (schoolbook rank 32)
    # Total rank = 49 + 24 + 32 = 105

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # c=0: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # c=1: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1

        # c=2: M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # c=3: M4 = A11(B01-B00)
        U[3][3] = 1
        V[0][3] = -1; V[1][3] = 1
        W[0][3] = 1; W[1][3] = 1

        # c=4: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # c=5: M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1

        # c=6: M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return U, V, W

    def strassen_4x4x4():
        """
        Build 4x4x4 Strassen decomposition from 7 2x2x2 blocks.
        Returns U, V, W of shape 16x49.
        """
        U2, V2, W2 = strassen_2x2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]

        # Strassen 4x4: split into 2x2 blocks
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]

        # 7 Strassen products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # For each of the 7 M's, use 2x2x2 Strassen (rank 7)
        # Total: 7 * 7 = 49

        # Define block offsets:
        # Block (bi, bj) of A: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # Block (bj, bk) of B: rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
        # Block (bi, bk) of C: rows 2*bi..2*bi+1, cols 2*bk..2*bk+1

        # Strassen product definitions:
        # (a_blocks, b_blocks, c_blocks)
        # Each block is (bi, bj) with sign
        strassen_defs = [
            # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
            ([((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)]),
            # M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
            ([((1,0), 1), ((1,1), 1)],
             [((0,0), 1)],
             [((1,0), 1), ((1,1), -1)]),
            # M3: A00(B10-B11) -> C01 += 1, C11 += 1
            ([((0,0), 1)],
             [((1,0), 1), ((1,1), -1)],
             [((0,1), 1), ((1,1), 1)]),
            # M4: A11(B01-B00) -> C00 += 1, C10 += 1
            ([((1,1), 1)],
             [((0,1), 1), ((0,0), -1)],
             [((0,0), 1), ((1,0), 1)]),
            # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
            ([((0,0), 1), ((0,1), 1)],
             [((1,1), 1)],
             [((0,0), -1), ((0,1), 1)]),
            # M6: (A10-A00)(B00+B10) -> C11 += 1
            ([((1,0), 1), ((0,0), -1)],
             [((0,0), 1), ((1,0), 1)],
             [((1,1), 1)]),
            # M7: (A01-A11)(B01+B11) -> C00 += 1
            ([((0,1), 1), ((1,1), -1)],
             [((0,1), 1), ((1,1), 1)],
             [((0,0), 1)]),
        ]

        for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
            for t in range(7):
                col = s * 7 + t

                # For each element in U2: U2[li*2+lj, t]
                # li, lj in {0,1} correspond to local row/col in 2x2 block
                for li in range(2):
                    for lj in range(2):
                        val_u = U2[li*2 + lj][t]
                        if val_u != 0:
                            # This coefficient applies to all blocks in a_blocks
                            for (bi, bj), sign_a in a_blocks:
                                i = 2*bi + li
                                j = 2*bj + lj
                                U[i*4 + j][col] += val_u * sign_a

                # For each element in V2: V2[lj*2+lk, t]
                for lj in range(2):
                    for lk in range(2):
                        val_v = V2[lj*2 + lk][t]
                        if val_v != 0:
                            for (bj, bk), sign_b in b_blocks:
                                j = 2*bj + lj
                                k = 2*bk + lk
                                V[j*4 + k][col] += val_v * sign_b

                # For each element in W2: W2[lk*2+li, t]
                for lk in range(2):
                    for li in range(2):
                        val_w = W2[lk*2 + li][t]
                        if val_w != 0:
                            for (bi, bk), sign_c in c_blocks:
                                i = 2*bi + li
                                k = 2*bk + lk
                                W[k*4 + i][col] += val_w * sign_c

        return U, V, W

    # Build the full 4x5x6 decomposition
    # Region A: i in [0,4), j in [0,4), k in [0,4) -> 4x4x4 Strassen (rank 49)
    # Region B: i in [0,4), j in [4,5), k in [0,6) -> 4x1x6 schoolbook (rank 24)
    # Region C: i in [0,4), j in [0,4), k in [4,6) -> 4x4x2 schoolbook (rank 32)

    U4, V4, W4 = strassen_4x4x4()
    rank_A = 49
    rank_B = 24
    rank_C = 32
    total_rank = rank_A + rank_B + rank_C

    U = [[0] * total_rank for _ in range(n * m)]  # 20 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x total_rank

    # Region A: 4x4x4 at i[0:4], j[0:4], k[0:4]
    for i in range(4):
        for j in range(4):
            for c in range(rank_A):
                U[i*5 + j][c] = U4[i*4 + j][c]
    for j in range(4):
        for k in range(4):
            for c in range(rank_A):
                V[j*6 + k][c] = V4[j*4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(rank_A):
                W[k*4 + i][c] = W4[k*4 + i][c]

    # Region B: 4x1x6 at i[0:4], j=4, k[0:6]
    col_offset = rank_A
    for i in range(4):
        for k in range(6):
            col = col_offset + i*6 + k
            U[i*5 + 4][col] = 1
            V[4*6 + k][col] = 1
            W[k*4 + i][col] = 1

    # Region C: 4x4x2 at i[0:4], j[0:4], k[4:6]
    col_offset = rank_A + rank_B
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                col = col_offset + (i*4 + j)*2 + (k-4)
                U[i*5 + j][col] = 1
                V[j*6 + k][col] = 1
                W[k*4 + i][col] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
