"""Matrix multiplication 4x5 by 5x6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    DEADLINE = 120.0

    def _elapsed():
        return time.time() - start_time

    def _deadline_exceeded():
        return _elapsed() > DEADLINE - 5.0

    # Strassen 2x2x2 decomposition
    # U, V, W indexed by (row_index, col) where row_index = i*m + j, etc.
    # For 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen terms (1-indexed for reference):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    def _strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication, each shape (4,7)."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # U rows: i*2+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V rows: j*2+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W rows: k*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1  # k=0,i=0 -> 0; k=1,i=1 -> 2
        # C00 += M1, C11 += M1

        # M2: (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # k=0,i=1 -> 2
        # C10 += M2

        # M3: A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1  # k=1,i=0 -> 1
        # C01 += M3, C11 += M3

        # M4: A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
        # Wait: W[k*2+i, c]. For C00: k=0,i=0 -> W[0,3]=1. For C10: k=0,i=1 -> W[2,3]=1.
        # But M4 contributes to both C00 and C10? No.
        # C00 = M1+M4-M5+M7, C10 = M2+M4
        # So M4 contributes to C00 (k=0,i=0) and C10 (k=0,i=1)? No, C10 is k=0,i=1.
        # Wait, C[i][k]. C10 means i=1,k=0. So W[k*2+i] = W[0*2+1] = W[1]? No, W[0*2+1]=W[1].
        # Let me re-check: W index is k*n+i where n=2. So W[k*2+i].
        # C00: i=0,k=0 -> W[0*2+0]=W[0]
        # C01: i=0,k=1 -> W[1*2+0]=W[2]
        # C10: i=1,k=0 -> W[0*2+1]=W[1]
        # C11: i=1,k=1 -> W[1*2+1]=W[3]

        # Let me redo this carefully.
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1: (A00+A11)(B00+B11) -> contributes to C00 and C11
        # U: A00 (i=0,j=0) -> U[0*2+0]=U[0], A11 (i=1,j=1) -> U[1*2+1]=U[3]
        U[0][0] = 1; U[3][0] = 1
        # V: B00 (j=0,k=0) -> V[0*2+0]=V[0], B11 (j=1,k=1) -> V[1*2+1]=V[3]
        V[0][0] = 1; V[3][0] = 1
        # W: C00 (i=0,k=0) -> W[0*2+0]=W[0], C11 (i=1,k=1) -> W[1*2+1]=W[3]
        W[0][0] = 1; W[3][0] = 1

        # M2: (A10+A11)B00 -> contributes to C10
        # U: A10 (i=1,j=0) -> U[1*2+0]=U[2], A11 (i=1,j=1) -> U[3]
        U[2][1] = 1; U[3][1] = 1
        # V: B00 (j=0,k=0) -> V[0]
        V[0][1] = 1
        # W: C10 (i=1,k=0) -> W[0*2+1]=W[1]
        W[1][1] = 1

        # M3: A00(B01-B11) -> contributes to C01 and C11
        # U: A00 (i=0,j=0) -> U[0]
        U[0][2] = 1
        # V: B01 (j=0,k=1) -> V[0*2+1]=V[1], B11 (j=1,k=1) -> V[3]
        V[1][2] = 1; V[3][2] = -1
        # W: C01 (i=0,k=1) -> W[1*2+0]=W[2], C11 (i=1,k=1) -> W[3]
        W[2][2] = 1; W[3][2] = 1

        # M4: A11(B10-B00) -> contributes to C00 and C10
        # U: A11 (i=1,j=1) -> U[3]
        U[3][3] = 1
        # V: B10 (j=1,k=0) -> V[1*2+0]=V[2], B00 (j=0,k=0) -> V[0]
        V[2][3] = 1; V[0][3] = -1
        # W: C00 (i=0,k=0) -> W[0], C10 (i=1,k=0) -> W[1]
        W[0][3] = 1; W[1][3] = 1

        # M5: (A00+A01)B11 -> contributes to C00 and C01
        # U: A00 (i=0,j=0) -> U[0], A01 (i=0,j=1) -> U[0*2+1]=U[1]
        U[0][4] = 1; U[1][4] = 1
        # V: B11 (j=1,k=1) -> V[3]
        V[3][4] = 1
        # W: C00 (i=0,k=0) -> W[0], C01 (i=0,k=1) -> W[2]
        W[0][4] = -1; W[2][4] = 1  # C00 = ...-M5+... so W[0] gets -1

        # M6: (A10-A00)(B00+B01) -> contributes to C11
        # U: A10 (i=1,j=0) -> U[2], A00 (i=0,j=0) -> U[0]
        U[2][5] = 1; U[0][5] = -1
        # V: B00 (j=0,k=0) -> V[0], B01 (j=0,k=1) -> V[1]
        V[0][5] = 1; V[1][5] = 1
        # W: C11 (i=1,k=1) -> W[3]
        W[3][5] = 1

        # M7: (A01-A11)(B10+B11) -> contributes to C00 and C01
        # U: A01 (i=0,j=1) -> U[1], A11 (i=1,j=1) -> U[3]
        U[1][6] = 1; U[3][6] = -1
        # V: B10 (j=1,k=0) -> V[2], B11 (j=1,k=1) -> V[3]
        V[2][6] = 1; V[3][6] = 1
        # W: C00 (i=0,k=0) -> W[0], C01 (i=0,k=1) -> W[2]
        W[0][6] = 1; W[2][6] = -1  # C00 = ...+M7, C01 = ...-M7? 
        # C00 = M1+M4-M5+M7, C01 = M3+M5
        # Wait, C01 doesn't include M7. Let me recheck Strassen.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M7 only contributes to C00.
        W[0][6] = 1

        return U, V, W

    # 2x2x3 decomposition (rank 11): A(2x2) * B(2x3) -> C(2x3)
    # Split B = [B1(2x2) | B2(2x1)]
    # C = A*B1 + A*B2
    # A*B1: 2x2x2, Strassen (7 terms)
    # A*B2: 2x2x1, schoolbook (4 terms)
    # Total: 11 terms
    # U shape (4, 11), V shape (6, 11), W shape (6, 11)
    # U[i*2+j, c], V[j*3+k, c], W[k*2+i, c]

    def _matmul_2x2x3():
        """Return (U, V, W) for 2x2 x 2x3 matrix multiplication, rank 11."""
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        # U rows: i*2+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V rows: j*3+k: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        # W rows: k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5

        # Part 1: A * B1 (2x2 x 2x2), B1 = B[:, 0:2]
        # Use Strassen on the 2x2x2 part
        # Strassen U,V,W for 2x2x2 (from above, adapted)
        # In the 2x2x3 context:
        # U indices same: i*2+j
        # V indices: j*3+k for k=0,1 (not 2)
        # W indices: k*2+i for k=0,1

        # M1: (A00+A11)(B00+B11) where B00=B[0,0], B11=B[1,1]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1  # B[0,0]=V[0*3+0]=V[0], B[1,1]=V[1*3+1]=V[4]
        W[0][0] = 1; W[3][0] = 1  # C[0,0]=W[0*2+0]=W[0], C[1,1]=W[1*2+1]=W[3]

        # M2: (A10+A11)B00 where B00=B[0,0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1  # C[1,0]=W[0*2+1]=W[1]

        # M3: A00(B01-B11) where B01=B[0,1], B11=B[1,1]
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1  # B[0,1]=V[0*3+1]=V[1], B[1,1]=V[4]
        W[2][2] = 1; W[3][2] = 1  # C[0,1]=W[1*2+0]=W[2], C[1,1]=W[3]

        # M4: A11(B10-B00) where B10=B[1,0], B00=B[0,0]
        U[3][3] = 1
        V[3][3] = 1; V[0][3] = -1  # B[1,0]=V[1*3+0]=V[3], B[0,0]=V[0]
        W[0][3] = 1; W[1][3] = 1  # C[0,0]=W[0], C[1,0]=W[1]

        # M5: (A00+A01)B11 where B11=B[1,1]
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1  # C[0,0] gets -M5, C[0,1] gets +M5

        # M6: (A10-A00)(B00+B01) where B00=B[0,0], B01=B[0,1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C[1,1]=W[3]

        # M7: (A01-A11)(B10+B11) where B10=B[1,0], B11=B[1,1]
        U[1][6] = 1; U[3][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1  # C[0,0]=W[0]

        # Part 2: A * B2 (2x2 x 2x1), B2 = B[:, 2]
        # Schoolbook: 4 terms
        # Term 7: A00*B20 -> C02
        U[0][7] = 1
        V[2][7] = 1  # B[0,2]=V[0*3+2]=V[2]
        W[4][7] = 1  # C[0,2]=W[2*2+0]=W[4]

        # Term 8: A01*B20 -> C02
        U[1][8] = 1
        V[5][8] = 1  # B[1,2]=V[1*3+2]=V[5]
        W[4][8] = 1

        # Term 9: A10*B20 -> C12
        U[2][9] = 1
        V[2][9] = 1
        W[5][9] = 1  # C[1,2]=W[2*2+1]=W[5]

        # Term 10: A11*B20 -> C12
        U[3][10] = 1
        V[5][10] = 1
        W[5][10] = 1

        return U, V, W

    # Now assemble 4x5 x 5x6
    # Split A (4x5) into 2x2 blocks vertically and horizontally:
    # A = [[A_LL (2x2), A_LR (2x3)],
    #      [A_RL (2x2), A_RR (2x3)]]
    # Split B (5x6) into 2x2 blocks:
    # B = [[B_TL (5x2), B_TR (5x4)]]
    # But we need 2x2 or 2x3 blocks in the middle dimension.
    # Let's split the middle dimension (5) into 2+3.
    # A = [A_L (4x2), A_R (4x3)]
    # B = [B_L (2x6), B_R (3x6)]
    # C = A_L * B_L + A_R * B_R
    # A_L * B_L: (4x2) * (2x6) -> split 4 into 2+2, 6 into 3+3
    #   A_L = [[A_LL (2x2), A_LR (2x3)],
    #          [A_RL (2x2), A_RR (2x3)]]  -- wait, A_L is 4x2
    #   Actually, let me think differently.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # j ranges 0..4. Split j = {0,1} and {2,3,4}.
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:5] @ B[2:5, :]
    # Term 1: (4x2) * (2x6)
    # Term 2: (4x3) * (3x6)
    #
    # For Term 1: (4x2)*(2x6). Split rows of A into 2+2, cols of B into 3+3.
    #   A1 = [[A1a (2x2)], [A1b (2x2)]]
    #   B1 = [[B1a (2x3), B1b (2x3)]]
    #   C1 = A1a*B1a + A1a*B1b + A1b*B1a + A1b*B1b
    #   Each is (2x2)*(2x3), rank 11. Total 4*11 = 44.
    #
    # For Term 2: (4x3)*(3x6). Split rows of A into 2+2, cols of B into 3+3.
    #   A2 = [[A2a (2x3)], [A2b (2x3)]]
    #   B2 = [[B2a (3x3), B2b (3x3)]]
    #   C2 = A2a*B2a + A2a*B2b + A2b*B2a + A2b*B2b
    #   Each is (2x3)*(3x3). 
    #   (2x3)*(3x3): Split B into 2x2 + 2x1? Or use a known decomposition.
    #   Actually, (2x3)*(3x3) can be seen as 2*(2x3)*(3x3) but that's not helpful.
    #   Let's split B2a (3x3) into [B2a_1 (3x2), B2a_2 (3x1)].
    #   A2a*B2a_1: (2x3)*(3x2). 
    #   (2x3)*(3x2): Split A into rows? A is 2x3. 
    #   This is getting complex. Let me use a simpler approach.
    #
    # Alternative: Use the fact that (4x5)*(5x6) can be decomposed by splitting
    # all three dimensions. But let me just use a block approach with known ranks.
    #
    # Simpler: Split 4=2+2, 5=2+3, 6=3+3.
    # C = A[0:2,0:2]*B[0:2,0:3] + A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:5]*B[2:5,0:3] + A[0:2,2:5]*B[2:5,3:6]
    #     + A[2:4,0:2]*B[0:2,0:3] + A[2:4,0:2]*B[0:2,3:6] + A[2:4,2:5]*B[2:5,0:3] + A[2:4,2:5]*B[2:5,3:6]
    #
    # Block types:
    # 1. (2x2)*(2x3): rank 11
    # 2. (2x3)*(3x3): need to find rank
    #
    # For (2x3)*(3x3): A(2x3), B(3x3). C(2x3).
    # Split B = [B1(3x2), B2(3x1)].
    # A*B1: (2x3)*(3x2). 
    # (2x3)*(3x2): A(2x3), B(3x2). C(2x2).
    # Split A = [A1(1x3), A2(1x3)]? No, A is 2x3.
    # Let's just use schoolbook for (2x3)*(3x3): 2*3*3 = 18 terms.
    # And (2x2)*(2x3): 11 terms (as computed).
    #
    # Total: 4 blocks of (2x2)*(2x3) = 4*11 = 44
    #        4 blocks of (2x3)*(3x3) = 4*18 = 72
    # Total rank = 44 + 72 = 116. Not great.
    #
    # Let me try (2x3)*(3x3) more carefully.
    # A(2x3), B(3x3). C[i,k] = sum_j A[i,j]*B[j,k], i=0,1; j=0,1,2; k=0,1,2.
    # This is the same as 2 separate (1x3)*(3x3) = (1x3) computations.
    # (1x3)*(3x3): rank = 3 (schoolbook). So (2x3)*(3x3) has rank <= 6.
    # Wait, that's just schoolbook: for each i (2 values), 3 terms. Total 6.
    # So rank of (2x3)*(3x3) is at most 6, and likely exactly 6.
    #
    # Let me verify: T[i,j,k] for (2x3)*(3x3). i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # This tensor has shape 2x3x3. The rank of the matrix multiplication tensor M_{n,m,p}
    # for n=2,m=3,p=3 is at most n*m*p = 18, but can be lower.
    # Actually, M_{2,3,3} is isomorphic to M_{3,2,3} by swapping i and j? No.
    # The tensor T[i,j,k] = 1 if we compute C[i,k] from A[i,j]*B[j,k].
    # This is just the standard matrix multiplication tensor. Its rank is known to be
    # at most n*m*p, but for small dimensions it can be lower.
    # For 2x3x3, I believe the rank is 11 (same as 2x2x3 by symmetry? No, 2x3x3 != 2x2x3).
    # Let me just use schoolbook: rank = 2*3*3 = 18.
    #
    # So total rank = 4*11 + 4*18 = 44 + 72 = 116.
    # This is worse than 120? No, 116 < 120. Good.
    #
    # Let me construct this.

    def _schoolbook_2x3x3():
        """Schoolbook for (2x3)*(3x3), rank 18."""
        U = [[0]*18 for _ in range(6)]  # i*3+j: 0..5
        V = [[0]*18 for _ in range(9)]  # j*3+k: 0..8
        W = [[0]*18 for _ in range(6)]  # k*2+i: 0..5
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W

    # Now assemble the full 4x5 x 5x6
    # Blocks:
    # A row split: [0:2], [2:4]
    # A col split / B row split: [0:2], [2:5]
    # B col split: [0:3], [3:6]
    #
    # 8 blocks:
    # (0,0): A[0:2,0:2] * B[0:2,0:3] -> (2x2)*(2x3), rank 11
    # (0,1): A[0:2,0:2] * B[0:2,3:6] -> (2x2)*(2x3), rank 11
    # (0,2): A[0:2,2:5] * B[2:5,0:3] -> (2x3)*(3x3), rank 18
    # (0,3): A[0:2,2:5] * B[2:5,3:6] -> (2x3)*(3x3), rank 18
    # (1,0): A[2:4,0:2] * B[0:2,0:3] -> (2x2)*(2x3), rank 11
    # (1,1): A[2:4,0:2] * B[0:2,3:6] -> (2x2)*(2x3), rank 11
    # (1,2): A[2:4,2:5] * B[2:5,0:3] -> (2x3)*(3x3), rank 18
    # (1,3): A[2:4,2:5] * B[2:5,3:6] -> (2x3)*(3x3), rank 18
    #
    # Total rank = 4*11 + 4*18 = 44 + 72 = 116

    U_2x2x3, V_2x2x3, W_2x2x3 = _matmul_2x2x3()
    U_2x3x3, V_2x3x3, W_2x3x3 = _schoolbook_2x3x3()

    total_rank = 116
    U = [[0]*total_rank for _ in range(20)]  # 4*5 = 20
    V = [[0]*total_rank for _ in range(30)]  # 5*6 = 30
    W = [[0]*total_rank for _ in range(24)]  # 6*4 = 24

    def _place_block(block_U, block_V, block_W, i_off, j_off, k_off, i_size, j_size, k_size, col_start):
        """Place a block decomposition into the global U, V, W."""
        rank_block = len(block_U[0])
        for i_local in range(i_size):
            for j_local in range(j_size):
                u_row = (i_off + i_local) * 5 + (j_off + j_local)
                for c in range(rank_block):
                    U[u_row][col_start + c] = block_U[i_local * j_size + j_local][c]
        for j_local in range(j_size):
            for k_local in range(k_size):
                v_row = (j_off + j_local) * 6 + (k_off + k_local)
                for c in range(rank_block):
                    V[v_row][col_start + c] = block_V[j_local * k_size + k_local][c]
        for k_local in range(k_size):
            for i_local in range(i_size):
                w_row = (k_off + k_local) * 4 + (i_off + i_local)
                for c in range(rank_block):
                    W[w_row][col_start + c] = block_W[k_local * i_size + i_local][c]

    col = 0
    # (0,0): A[0:2,0:2] * B[0:2,0:3]
    _place_block(U_2x2x3, V_2x2x3, W_2x2x3, 0, 0, 0, 2, 2, 3, col)
    col += 11

    # (0,1): A[0:2,0:2] * B[0:2,3:6]
    _place_block(U_2x2x3, V_2x2x3, W_2x2x3, 0, 0, 3, 2, 2, 3, col)
    col += 11

    # (0,2): A[0:2,2:5] * B[2:5,0:3]
    _place_block(U_2x3x3, V_2x3x3, W_2x3x3, 0, 2, 0, 2, 3, 3, col)
    col += 18

    # (0,3): A[0:2,2:5] * B[2:5,3:6]
    _place_block(U_2x3x3, V_2x3x3, W_2x3x3, 0, 2, 3, 2, 3, 3, col)
    col += 18

    # (1,0): A[2:4,0:2] * B[0:2,0:3]
    _place_block(U_2x2x3, V_2x2x3, W_2x2x3, 2, 0, 0, 2, 2, 3, col)
    col += 11

    # (1,1): A[2:4,0:2] * B[0:2,3:6]
    _place_block(U_2x2x3, V_2x2x3, W_2x2x3, 2, 0, 3, 2, 2, 3, col)
    col += 11

    # (1,2): A[2:4,2:5] * B[2:5,0:3]
    _place_block(U_2x3x3, V_2x3x3, W_2x3x3, 2, 2, 0, 2, 3, 3, col)
    col += 18

    # (1,3): A[2:4,2:5] * B[2:5,3:6]
    _place_block(U_2x3x3, V_2x3x3, W_2x3x3, 2, 2, 3, 2, 3, 3, col)
    col += 18

    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
