def construct():
    import time
    import numpy as np

    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin

    n, m, p = 4, 4, 8

    # Rank-49 bilinear algorithm for 4x4 matrix multiplication
    # Derived from hierarchical application of Strassen's algorithm
    # C = A * B where A, B, C are 4x4
    #
    # Strassen 2x2: M1..M7 products, 7 instead of 8
    # For 4x4: split into 2x2 blocks, apply Strassen to get 7*(4/2)^2 = 28? No.
    # Actually for 4x4, we split into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Each block is 2x2.
    # Strassen gives:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B12-B22)
    #   M4 = A22(B21-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For 4x4, each "entry" in the 2x2 block structure is a 2x2 matrix.
    # The products M1..M7 are 2x2 matrix multiplications.
    # If we use schoolbook for the 2x2 multiplications, each M costs 4 scalar mults.
    # Total: 7 * 4 = 28 scalar multiplications. That's Laderman's rank!
    #
    # Wait, that's exactly Laderman. So rank-28 for 4x4.
    # For 4x4x8: split B into two 4x4 blocks, use rank-28 for each.
    # Total rank = 56.
    #
    # But the strategy says "rank-49 bilinear algorithm". Let me think again.
    # 
    # Actually, there's a different way to apply Strassen hierarchically.
    # If we treat 4x4 as a 2x2 block matrix where each block is 2x2,
    # and apply Strassen at the block level, we get 7 block-matrix multiplications.
    # Each block multiplication (2x2 by 2x2) can be done with 4 scalar mults (schoolbook)
    # giving 7*4 = 28. That's Laderman.
    #
    # Alternatively, if we could do 2x2 multiplication with rank 3 (Strassen),
    # then 7 * 3 = 21. But that's not integer/half-integer in the same way.
    #
    # Hmm, the "rank-49" might come from a different construction.
    # Let me think: 4x4x4 has known rank 28 (Laderman). 
    # 4x4x8 = two independent 4x4x4 problems, so rank <= 56.
    #
    # The strategy mentions "rank-49 bilinear algorithm for 4x4 matrix multiplication".
    # This might be a different, less optimal algorithm. But 49 > 28, so it's worse.
    # Unless it's for a different problem structure.
    #
    # Wait, maybe the 49 comes from: for 4x4, using a 7x7 Strassen-like scheme
    # where each of the 7 products involves a 2x2 multiplication done with 7 terms?
    # No, that would be 7*7=49. But 2x2 multiplication has rank 7 (Strassen),
    # so 7 Strassen products each of rank 7 gives 49.
    #
    # So: 4x4 = 2x2 blocks, 7 Strassen products, each 2x2x2x2 with rank 7.
    # Total rank = 7 * 7 = 49 for 4x4x4.
    # For 4x4x8: 49 * 2 = 98.
    #
    # This is worse than Laderman (28*2=56), but let me implement it as specified.
    # Actually, the directive says to use this approach. Let me implement it.
    #
    # Strassen for 2x2:
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
    #
    # For 4x4x4:
    # Split A into 4 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # Split B into 4 blocks of 2x2: B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen at block level:
    # P1 = (A00+A11)(B00+B11)  -> 2x2x2x2 multiplication, rank 7
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # Each Pi is a 2x2 matrix computed from a 2x2x2x2 multiplication.
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    #
    # Total rank: 7 (block-level) * 7 (each 2x2x2x2) = 49.
    #
    # For 4x4x8: B is 4x8, split into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left, C_right = A * B_right.
    # Each uses rank 49, total rank = 98.

    # Build the rank-49 decomposition for 4x4x4
    # Then extend to 4x4x8 by doubling.

    # First, let's build the 2x2x2x2 Strassen decomposition (rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # A2 is 2x2: indices 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B2 is 2x2: indices 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    # C2 is 2x2: indices W[k*2+i]: 0=C[0,0], 1=C[1,0], 2=C[0,1], 3=C[1,1]
    # Wait, the convention is W[k*n+i,r]. For 2x2, n=2.
    # W[0*2+0]=C[0,0], W[1*2+0]=C[1,0], W[0*2+1]=C[0,1], W[1*2+1]=C[1,1]
    # So C index: k*2+i where i is row of A, k is col of B / row of C.
    # C[i,k] is at index k*2+i.
    # C[0,0]: k=0,i=0 -> 0
    # C[1,0]: k=0,i=1 -> 1
    # C[0,1]: k=1,i=0 -> 2
    # C[1,1]: k=1,i=1 -> 3

    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> indices 0,3
    # V: B00=1, B11=1 -> indices 0,3
    # W: contributes to C00, C01, C10, C11 with coeff 1
    # C[0,0]=M1, C[0,1]=M1, C[1,0]=M1, C[1,1]=M1? No.
    # C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    # So M1 contributes +1 to C00 and +1 to C11.
    # C[0,0] index 0, C[1,1] index 3.
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    # M2 = (A10+A11)B00
    # U: A10=1, A11=1 -> indices 2,3
    # V: B00=1 -> index 0
    # W: M2 contributes +1 to C10, -1 to C11
    # C[1,0] index 1, C[1,1] index 3
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1

    # M3 = A00(B01-B11)
    # U: A00=1 -> index 0
    # V: B01=1, B11=-1 -> indices 1,3
    # W: M3 contributes +1 to C01, +1 to C11
    # C[0,1] index 2, C[1,1] index 3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1

    # M4 = A11(B10-B00)
    # U: A11=1 -> index 3
    # V: B10=1, B00=-1 -> indices 2,0
    # W: M4 contributes +1 to C00, +1 to C10
    # C[0,0] index 0, C[1,0] index 1
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1

    # M5 = (A00+A01)B11
    # U: A00=1, A01=1 -> indices 0,1
    # V: B11=1 -> index 3
    # W: M5 contributes -1 to C00, +1 to C01
    # C[0,0] index 0, C[0,1] index 2
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1

    # M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> indices 2,0
    # V: B00=1, B01=1 -> indices 0,1
    # W: M6 contributes +1 to C11
    # C[1,1] index 3
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1

    # M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> indices 1,3
    # V: B10=1, B11=1 -> indices 2,3
    # W: M7 contributes +1 to C00
    # C[0,0] index 0
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Verify 2x2x2x2 Strassen
    U2_np = np.array(U2, dtype=float)  # (4, 7)
    V2_np = np.array(V2, dtype=float)  # (4, 7)
    W2_np = np.array(W2, dtype=float)  # (4, 7)

    np.random.seed(42)
    A2_test = np.random.rand(2, 2)
    B2_test = np.random.rand(2, 2)
    C2_test = A2_test @ B2_test

    a2_vec = A2_test.flatten()  # (4,)
    b2_vec = B2_test.flatten()  # (4,)

    left2 = U2_np.T @ a2_vec  # (7,)
    right2 = V2_np.T @ b2_vec  # (7,)
    prods2 = left2 * right2  # (7,)
    C2_recon = (W2_np.T @ prods2).reshape(2, 2)

    err2 = np.max(np.abs(C2_test - C2_recon))
    if err2 > 1e-10:
        # Fall back to schoolbook for 4x4x8
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

    # Build rank-49 decomposition for 4x4x4
    # 7 Strassen block products, each of rank 7 (2x2x2x2)
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    # A4: 4x4, split into 2x2 blocks:
    #   A00 = A[0:2,0:2] -> indices 0,1,4,5
    #   A01 = A[0:2,2:4] -> indices 2,3,6,7
    #   A10 = A[2:4,0:2] -> indices 8,9,12,13
    #   A11 = A[2:4,2:4] -> indices 10,11,14,15
    # B4: 4x4, split into 2x2 blocks:
    #   B00 = B[0:2,0:2] -> indices 0,1,4,5
    #   B01 = B[0:2,2:4] -> indices 2,3,6,7
    #   B10 = B[2:4,0:2] -> indices 8,9,12,13
    #   B11 = B[2:4,2:4] -> indices 10,11,14,15
    # C4: 4x4, W index k*4+i:
    #   C00 = C[0:2,0:2] -> indices 0,1,4,5
    #   C01 = C[0:2,2:4] -> indices 8,9,12,13
    #   C10 = C[2:4,0:2] -> indices 2,3,6,7
    #   C11 = C[2:4,2:4] -> indices 10,11,14,15

    # Block indices for 4x4 matrices
    # A block (row_block, col_block) -> 4 indices
    def a_block_idx(r, c):
        return [r*4 + c, r*4 + c + 1, r*4 + 4 + c, r*4 + 4 + c + 1]

    def b_block_idx(r, c):
        return [r*4 + c, r*4 + c + 1, r*4 + 4 + c, r*4 + 4 + c + 1]

    def c_block_idx(r, c):
        # C[r*2:(r+1)*2, c*2:(c+1)*2]
        # W index: k*4+i where i is row, k is col
        # C[i,k] for i in r*2..r*2+1, k in c*2..c*2+1
        idx = []
        for i in range(r*2, r*2+2):
            for k in range(c*2, c*2+2):
                idx.append(k*4 + i)
        return idx

    # 7 Strassen products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)

    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6

    # Each Pi is a 2x2x2x2 multiplication with rank 7.
    # For each Pi, we need to map the 2x2 block entries to the 4x4 indices.

    # P1: left = A00+A11, right = B00+B11, output -> C00, C11
    # A00 indices: a_block_idx(0,0) = [0,1,4,5]
    # A11 indices: a_block_idx(1,1) = [10,11,14,15]
    # B00 indices: b_block_idx(0,0) = [0,1,4,5]
    # B11 indices: b_block_idx(1,1) = [10,11,14,15]
    # C00 indices: c_block_idx(0,0) = [0,1,4,5]
    # C11 indices: c_block_idx(1,1) = [10,11,14,15]

    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]

    col = 0
    for pi in range(7):
        # Determine left, right, output blocks and signs
        if pi == 0:  # P1
            left_blocks = [(0,0,1), (1,1,1)]
            right_blocks = [(0,0,1), (1,1,1)]
            out_blocks = [(0,0,1), (1,1,1)]
        elif pi == 1:  # P2
            left_blocks = [(1,0,1), (1,1,1)]
            right_blocks = [(0,0,1)]
            out_blocks = [(1,0,1), (1,1,-1)]
        elif pi == 2:  # P3
            left_blocks = [(0,0,1)]
            right_blocks = [(0,1,1), (1,1,-1)]
            out_blocks = [(0,1,1), (1,1,1)]
        elif pi == 3:  # P4
            left_blocks = [(1,1,1)]
            right_blocks = [(1,0,1), (0,0,-1)]
            out_blocks = [(0,0,1), (1,0,1)]
        elif pi == 4:  # P5
            left_blocks = [(0,0,1), (0,1,1)]
            right_blocks = [(1,1,1)]
            out_blocks = [(0,0,-1), (0,1,1)]
        elif pi == 5:  # P6
            left_blocks = [(1,0,1), (0,0,-1)]
            right_blocks = [(0,0,1), (0,1,1)]
            out_blocks = [(1,1,1)]
        elif pi == 6:  # P7
            left_blocks = [(0,1,1), (1,1,-1)]
            right_blocks = [(1,0,1), (1,1,1)]
            out_blocks = [(0,0,1)]

        # For each of the 7 Strassen 2x2x2x2 terms
        for si in range(7):
            c_idx = col + si
            # Left factor: sum of U2[si] * A_block_entry
            for (br, bc, sign) in left_blocks:
                for local_i in range(4):
                    a_idx = a_block_idx(br, bc)[local_i]
                    U4[a_idx][c_idx] += sign * U2[local_i][si]

            # Right factor: sum of V2[si] * B_block_entry
            for (br, bc, sign) in right_blocks:
                for local_i in range(4):
                    b_idx = b_block_idx(br, bc)[local_i]
                    V4[b_idx][c_idx] += sign * V2[local_i][si]

            # Output factor: sum of W2[si] * C_block_entry
            for (br, bc, sign) in out_blocks:
                for local_i in range(4):
                    c_idx2 = c_block_idx(br, bc)[local_i]
                    W4[c_idx2][c_idx] += sign * W2[local_i][si]

        col += 7

    # Verify 4x4x4 rank-49 decomposition
    U4_np = np.array(U4, dtype=float)
    V4_np = np.array(V4, dtype=float)
    W4_np = np.array(W4, dtype=float)

    A4_test = np.random.rand(4, 4)
    B4_test = np.random.rand(4, 4)
    C4_test = A4_test @ B4_test

    a4_vec = A4_test.flatten()
    b4_vec = B4_test.flatten()

    left4 = U4_np.T @ a4_vec  # (49,)
    right4 = V4_np.T @ b4_vec  # (49,)
    prods4 = left4 * right4  # (49,)
    C4_recon = (W4_np.T @ prods4).reshape(4, 4)

    err4 = np.max(np.abs(C4_test - C4_recon))
    if err4 > 1e-10:
        # Fall back to schoolbook
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

    # Extend to 4x4x8: B is 4x8, split into B_left (4x4) and B_right (4x4)
    # U: (16, 98) - first 49 cols = U4, next 49 cols = U4
    # V: (32, 98) - first 49 cols: V4 in rows 0-15, zeros in rows 16-31
    #               next 49 cols: zeros in rows 0-15, V4 in rows 16-31
    # W: (32, 98) - first 49 cols: W4 in rows 0-15, zeros in rows 16-31
    #               next 49 cols: zeros in rows 0-15, W4 in rows 16-31

    rank_total = 98
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(32)]
    W = [[0] * rank_total for _ in range(32)]

    for r in range(49):
        for i in range(16):
            U[i][r] = U4[i][r]
            V[i][r] = V4[i][r]
            W[i][r] = W4[i][r]

            U[i][r + 49] = U4[i][r]
            V[i + 16][r + 49] = V4[i][r]
            W[i + 16][r + 49] = W4[i][r]

    return {"U": U, "V": V, "W": W}
