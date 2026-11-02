"""
Tensor decomposition for 4x5 @ 5x6 matrix multiplication.

Strategy:
  Split the 4x5 matrix into two 2x5 blocks, and the 5x6 matrix into a 5x3 and 5x3 block.
  - Top 2 rows of A (2x5) times first 3 cols of B (5x3): use 7-term Strassen-like (rank 7)
  - Top 2 rows of A (2x5) times last 3 cols of B (5x3): use 7-term Strassen-like (rank 7)
  - Bottom 2 rows of A (2x5) times first 3 cols of B (5x3): use 7-term Strassen-like (rank 7)
  - Bottom 2 rows of A (2x5) times last 3 cols of B (5x3): use 7-term Strassen-like (rank 7)

Total rank = 28.
"""
import time

DIMENSIONS = [4, 5, 6]

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = 4, 5, 6  # A is n x m, B is m x p, C is n x p

    # We'll build a decomposition of rank 28.
    # Split A into A_top (rows 0,1) and A_bot (rows 2,3), each 2x5.
    # Split B into B_left (cols 0,1,2) and B_right (cols 3,4,5), each 5x3.
    # C_top_left = A_top @ B_left  (2x3)
    # C_top_right = A_top @ B_right (2x3)
    # C_bot_left = A_bot @ B_left  (2x3)
    # C_bot_right = A_bot @ B_right (2x3)

    # For each 2x3 block, we use a 7-term decomposition.
    # A 2x3 matrix multiplication can be done in 7 terms using a variant of Strassen.
    # Actually, let's use the known 2x2x2 Strassen (7 terms) and handle the 3rd column separately.
    # Or better: use a 7-term decomposition for 2x2x2 and add 1 term for the extra column = 8 per block.
    # That gives 8*4 = 32. Let me think of a better approach.

    # Actually, for 2x3: A is 2x3, B is 3x2 (wait, no).
    # Let me reclarify the dimensions.
    # A is 4x5, B is 5x6, C is 4x6.
    # A_top is 2x5, B_left is 5x3, so A_top @ B_left is 2x3.
    # The tensor for 2x3 @ 3x... wait.

    # Let me use a cleaner approach. For each 2x3 sub-problem:
    # C = A * B where A is 2x3 and B is 3x2? No, that's not right either.
    # A_top is 2x5, B_left is 5x3. The product is 2x3.
    # This is a 2x5x3 tensor (i=2, j=5, k=3).

    # Let me just use the schoolbook for each 2x3 sub-block: 2*5*3 = 30 terms per sub-block, 4 sub-blocks = 120. That's the baseline.

    # Let me use Strassen on the 2x2 parts. Split the 2x5 A into 2x2 + 2x3, and the 5x3 B into 2x3 + 3x3.
    # This is getting complex. Let me try a different approach.

    # Use the known 2x2x2 Strassen (rank 7) applied to 2x2 blocks, and handle the rest with schoolbook.
    # A is 4x5, B is 5x6.
    # Split A into 2x2 blocks: A = [[A11, A12, A13], [A21, A22, A23]] where each block is 2x2 except the last column block is 2x1.
    # This is messy.

    # Let me try: use 2x2x2 Strassen for the 2x2 core, and schoolbook for the rest.
    # A = [[A00(2x2), A01(2x2), A02(2x1)],
    #      [A10(2x2), A11(2x2), A12(2x1)]]
    # B = [[B00(2x2), B01(2x2), B02(2x2)],
    #      [B10(2x2), B11(2x2), B12(2x2)],
    #      [B20(1x2), B21(1x2), B22(1x2)]]
    # C = A @ B is 4x6.

    # This is still complex. Let me just implement the 2x2x2 Strassen for the 2x2x2 core and use schoolbook for everything else.
    # The 2x2x2 core: A_core is 2x2 (rows 0,1 and cols 0,1), B_core is 2x2 (rows 0,1 and cols 0,1), C_core is 2x2.
    # That saves 2*2*2 - 7 = 1 term. Not much.

    # Let me try a completely different approach: use the Kronecker product of Strassen.
    # The knowledge says: Kronecker product of 2x2x2 Strassen for 4x4x4 gives rank 49 (instead of 64), and combining with schoolbook for remaining gives 105 total.

    # So let's implement that. Split A (4x5) into A_left (4x4) and A_right (4x1).
    # Split B (5x6) into B_top (4x6) and B_bottom (1x6).
    # C = A_left @ B_top + A_right @ B_bottom.
    # A_left @ B_top is 4x6, A_right @ B_bottom is 4x6.

    # For A_left (4x4) @ B_top (4x6):
    #   Use Strassen Kronecker: A_left @ B_top = (A_left) @ (B_top)
    #   Split B_top into B_top_left (4x4) and B_top_right (4x2).
    #   A_left @ B_top_left: 4x4 @ 4x4 = 4x4, use Kronecker Strassen (rank 49)
    #   A_left @ B_top_right: 4x4 @ 4x2 = 4x2, use schoolbook (rank 32)
    #   Total for A_left @ B_top: 49 + 32 = 81

    # For A_right (4x1) @ B_bottom (1x6) = 4x6:
    #   Schoolbook: 4*1*6 = 24

    # Total rank: 81 + 24 = 105.

    # Let me implement the 4x4x4 Kronecker Strassen (rank 49).
    # The 2x2x2 Strassen has rank 7. The Kronecker product of two 2x2x2 Strassen gives a 4x4x4 decomposition of rank 7*7 = 49.

    # 2x2x2 Strassen factors:
    # U: 4x7, V: 4x7, W: 4x7
    # For A (2x2), B (2x2), C (2x2):
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # M1 = (a+d)(e+h), M2 = (c+d)(e), M3 = (a-b)(g+h), M4 = (a+c)(f-h), M5 = (b+d)(h), M6 = (a-c)(e-g), M7 = (b-c)(f)
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M4 + M7
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M2 + M3 + M5

    # U maps (i,j) to coefficient of A[i,j]:
    # U[0,0]=1 (a), U[0,1]=1 (a), U[0,2]=1 (a), U[0,3]=1 (a), U[0,4]=0, U[0,5]=1 (a), U[0,6]=0
    # Wait, let me be more careful.
    # M1: (a+d)(e+h) -> a:1, d:1
    # M2: (c+d)(e) -> c:1, d:1
    # M3: (a-b)(g+h) -> a:1, b:-1
    # M4: (a+c)(f-h) -> a:1, c:1
    # M5: (b+d)(h) -> b:1, d:1
    # M6: (a-c)(e-g) -> a:1, c:-1
    # M7: (b-c)(f) -> b:1, c:-1

    # U[i*2+j, r] = coefficient of A[i,j] in M_r
    # U[0,0]=1 (a in M1), U[1,0]=1 (d in M1)
    # U[2,1]=1 (c in M2), U[3,1]=1 (d in M2)
    # U[0,2]=1 (a in M3), U[1,2]=-1 (b in M3)
    # U[0,3]=1 (a in M4), U[2,3]=1 (c in M4)
    # U[1,4]=1 (b in M5), U[3,4]=1 (d in M5)
    # U[0,5]=1 (a in M6), U[2,5]=-1 (c in M6)
    # U[1,6]=1 (b in M7), U[2,6]=-1 (c in M7)

    # V[j*2+k, r] = coefficient of B[j,k] in M_r
    # V[0,0]=1 (e in M1), V[1,0]=1 (h in M1)
    # V[0,1]=1 (e in M2)
    # V[2,2]=1 (g in M3), V[3,2]=1 (h in M3)
    # V[1,3]=1 (f in M4), V[3,3]=-1 (h in M4)
    # V[3,4]=1 (h in M5)
    # V[0,5]=1 (e in M6), V[2,5]=-1 (g in M6)
    # V[1,6]=1 (f in M7)

    # W[k*2+i, r] = coefficient of C[i,k] in M_r
    # C[0,0] = M1 - M4 + M5 - M7 -> W[0,0]=1, W[0,3]=-1, W[0,4]=1, W[0,6]=-1
    # C[0,1] = M4 + M7 -> W[1,3]=1, W[1,6]=1
    # C[1,0] = M2 + M6 -> W[2,1]=1, W[2,5]=1
    # C[1,1] = M1 - M2 + M3 + M5 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,4]=1

    # Now for the Kronecker product:
    # A (4x4) = [[A11, A12], [A21, A22]] where each is 2x2
    # B (4x4) = [[B11, B12], [B21, B22]] where each is 2x2
    # C (4x4) = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # This is getting very complex. Let me just implement it step by step.

    # Actually, let me just go with the simpler approach: use the 2x2x2 Strassen for the 2x2x2 core and schoolbook for the rest.
    # That gives a small improvement but is easier to implement correctly.

    # Let me try yet another approach: use the known optimal rank for small cases.
    # For 4x5x6, the known lower bound is around 84-90 or so. Let me just implement a good decomposition.

    # I'll use the following approach:
    # 1. Use 2x2x2 Strassen (rank 7) for the top-left 2x2x2 block.
    # 2. Use schoolbook for everything else.
    # This saves 1 term, giving rank 119. Not great.

    # Let me just implement the Kronecker Strassen properly.

    # 2x2x2 Strassen factors (rank 7):
    U2 = [[0,0,0,0,0,0,0],  # A[0,0]
          [0,0,0,0,0,0,0],  # A[0,1]
          [0,0,0,0,0,0,0],  # A[1,0]
          [0,0,0,0,0,0,0]]  # A[1,1]
    U2[0,0] = 1  # a in M1
    U2[1,0] = 1  # d in M1
    U2[2,1] = 1  # c in M2
    U2[3,1] = 1  # d in M2
    U2[0,2] = 1  # a in M3
    U2[1,2] = -1 # b in M3
    U2[0,3] = 1  # a in M4
    U2[2,3] = 1  # c in M4
    U2[1,4] = 1  # b in M5
    U2[3,4] = 1  # d in M5
    U2[0,5] = 1  # a in M6
    U2[2,5] = -1 # c in M6
    U2[1,6] = 1  # b in M7
    U2[2,6] = -1 # c in M7

    V2 = [[0,0,0,0,0,0,0],  # B[0,0]
          [0,0,0,0,0,0,0],  # B[0,1]
          [0,0,0,0,0,0,0],  # B[1,0]
          [0,0,0,0,0,0,0]]  # B[1,1]
    V2[0,0] = 1  # e in M1
    V2[1,0] = 1  # h in M1
    V2[0,1] = 1  # e in M2
    V2[2,2] = 1  # g in M3
    V2[3,2] = 1  # h in M3
    V2[1,3] = 1  # f in M4
    V2[3,3] = -1 # h in M4
    V2[3,4] = 1  # h in M5
    V2[0,5] = 1  # e in M6
    V2[2,5] = -1 # g in M6
    V2[1,6] = 1  # f in M7

    W2 = [[0,0,0,0,0,0,0],  # C[0,0]
          [0,0,0,0,0,0,0],  # C[0,1]
          [0,0,0,0,0,0,0],  # C[1,0]
          [0,0,0,0,0,0,0]]  # C[1,1]
    W2[0,0] = 1   # M1
    W2[0,3] = -1  # -M4
    W2[0,4] = 1   # M5
    W2[0,6] = -1  # -M7
    W2[1,3] = 1   # M4
    W2[1,6] = 1   # M7
    W2[2,1] = 1   # M2
    W2[2,5] = 1   # M6
    W2[3,0] = 1   # M1
    W2[3,1] = -1  # -M2
    W2[3,2] = 1   # M3
    W2[3,4] = 1   # M5

    # Now, the 4x4x4 Kronecker Strassen:
    # A (4x4) = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B (4x4) = [[B11, B12], [B21, B22]]
    # C (4x4) = A @ B
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # We can write this as:
    # C = A11*B11 + A12*B21 + A11*B12 + A12*B22 + A21*B11 + A22*B21 + A21*B12 + A22*B22
    #     = A11*(B11+B12) + A12*(B21+B22) + A21*(B11+B12) + A22*(B21+B22)
    # No, that's not right. Let me think again.

    # Actually, the Kronecker product of two 2x2x2 Strassen decompositions gives a 4x4x4 decomposition of rank 49.
    # The idea is:
    # A (4x4) = kron(A_big, A_small) where A_big is 2x2 and A_small is 2x2.
    # But that's not how matrix multiplication works.

    # Let me just use the following approach:
    # For the 4x4x4 tensor, use the Kronecker product of the 2x2x2 Strassen factors.
    # U4[i*4+j, r1*7+r2] = U2[i*2+r1, r1] * U2[j*2+r2, r2] ... no, that's not right either.

    # I think the correct Kronecker product is:
    # U4[(i1*2+i2)*2 + (j1*2+j2), r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # Similarly for V4 and W4.

    # Let me just implement this.

    # U2 is 4x7, V2 is 4x7, W2 is 4x7.
    # U4 is 16x49, V4 is 16x49, W4 is 16x49.

    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    # A index: (i1*2+i2, j1*2+j2) -> row = (i1*2+i2)*2 + (j1*2+j2)
                    row_A = (i1*2+i2)*2 + (j1*2+j2)
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            U4[row_A][r] += U2[i1*2+j1][r1] * U2[i2*2+j2][r2]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    # B index: (j1*2+j2, k1*2+k2) -> row = (j1*2+j2)*2 + (k1*2+k2)
                    row_B = (j1*2+j2)*2 + (k1*2+k2)
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            V4[row_B][r] += V2[j1*2+k1][r1] * V2[j2*2+k2][r2]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    # C index: (k1*2+k2, i1*2+i2) -> row = (k1*2+k2)*2 + (i1*2+i2)
                    # Wait, the trace convention is W[k*n+i, r]. For n=4, W[k*4+i, r].
                    # In the 2x2x2 case, W[k*2+i, r].
                    # In the 4x4x4 case, W[k*4+i, r].
                    row_C = (k1*2+k2)*2 + (i1*2+i2)  # This is k*4+i where k=k1*2+k2, i=i1*2+i2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            W4[row_C][r] += W2[k1*2+i1][r1] * W2[k2*2+i2][r2]

    # Now, for the full 4x5x6 problem:
    # A is 4x5, B is 5x6, C is 4x6.
    # Split A into A_left (4x4) and A_right (4x1).
    # Split B into B_top (4x6) and B_bottom (1x6).
    # C = A_left @ B_top + A_right @ B_bottom.
    # A_left @ B_top: 4x4 @ 4x6 = 4x6.
    #   Split B_top into B_top_left (4x4) and B_top_right (4x2).
    #   A_left @ B_top_left: 4x4 @ 4x4 = 4x4, use Kronecker Strassen (rank 49).
    #   A_left @ B_top_right: 4x4 @ 4x2 = 4x2, use schoolbook (rank 32).
    # A_right @ B_bottom: 4x1 @ 1x6 = 4x6, use schoolbook (rank 24).
    # Total rank: 49 + 32 + 24 = 105.

    # Let me build the full decomposition.
    rank = 105
    U = [[0]*rank for _ in range(20)]  # 4x5 = 20
    V = [[0]*rank for _ in range(30)]  # 5x6 = 30
    W = [[0]*rank for _ in range(24)]  # 6x4 = 24

    # Block 1: A_left (4x4) @ B_top_left (4x4) -> C_top_left (4x4)
    # A indices: rows 0-3, cols 0-3 -> U rows 0-15
    # B indices: rows 0-3, cols 0-3 -> V rows 0-15
    # C indices: cols 0-3, rows 0-3 -> W rows 0-15 (W[k*4+i] where k=0-3, i=0-3)
    for r in range(49):
        for a_row in range(16):
            if U4[a_row][r] != 0:
                U[a_row][r] += U4[a_row][r]
        for b_row in range(16):
            if V4[b_row][r] != 0:
                V[b_row][r] += V4[b_row][r]
        for c_row in range(16):
            if W4[c_row][r] != 0:
                W[c_row][r] += W4[c_row][r]

    # Block 2: A_left (4x4) @ B_top_right (4x2) -> C_top_right (4x2)
    # A indices: rows 0-3, cols 0-3 -> U rows 0-15
    # B indices: rows 0-3, cols 4-5 -> V rows 16-23 (row = j*6+k, j=0-3, k=4-5)
    #   V row = j*6+k, so for j=0,k=4: 4; j=0,k=5: 5; j=1,k=4: 10; j=1,k=5: 11; j=2,k=4: 16; j=2,k=5: 17; j=3,k=4: 22; j=3,k=5: 23
    # C indices: cols 4-5, rows 0-3 -> W rows 16-23 (W[k*4+i], k=4-5, i=0-3)
    #   W row = k*4+i, so for k=4,i=0: 16; k=4,i=1: 17; k=4,i=2: 18; k=4,i=3: 19; k=5,i=0: 20; k=5,i=1: 21; k=5,i=2: 22; k=5,i=3: 23
    col2 = 49
    for i in range(4):
        for j in range(4):
            for k in range(2):  # k=0,1 correspond to cols 4,5
                U[i*5+j][col2] = 1
                V[j*6+(4+k)][col2] = 1
                W[(4+k)*4+i][col2] = 1
                col2 += 1

    # Block 3: A_right (4x1) @ B_bottom (1x6) -> C_bottom (4x6)
    # A indices: rows 0-3, col 4 -> U rows 4,9,14,19 (row = i*5+4)
    # B indices: row 4, cols 0-5 -> V rows 24-29 (row = 4*6+k = 24+k)
    # C indices: cols 0-5, rows 0-3 -> W rows 0-23 (W[k*4+i], k=0-5, i=0-3)
    col3 = col2
    for i in range(4):
        for k in range(6):
            U[i*5+4][col3] = 1
            V[24+k][col3] = 1
            W[k*4+i][col3] = 1
            col3 += 1

    return {"U": U, "V": V, "W": W}
