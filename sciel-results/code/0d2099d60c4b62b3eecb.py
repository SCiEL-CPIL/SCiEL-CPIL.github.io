"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=4, m=4, p=8

    # Build the multiplication tensor T[a, b, c]
    # a = i*m + j (index into A: n x m = 4x4 = 16)
    # b = j*p + k (index into B: m x p = 4x8 = 32)
    # c = k*n + i (index into C: p x n = 8x4 = 32)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook rank = n*m*p = 128
    schoolbook_rank = n * m * p

    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Try to find a better decomposition using Strassen-like block structure
    # For 4x4 x 4x8: split B into two 4x4 blocks (columns 0-3 and 4-7)
    # This is equivalent to two independent 4x4x4 multiplications
    # Best known rank for 4x4x4: 49 (using Strassen's 7 for 2x2x2 blocks recursively)
    # So total rank should be 2*49 = 98

    # Strassen's algorithm for 2x2x2: rank 7
    # The 7 tensor factors for 2x2x2 matrix multiplication:
    # Given A (2x2), B (2x2), compute C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # Strassen's 7 products:
    # P1 = (a+b)*(e+f)
    # P2 = (c+d)*e
    # P3 = a*(f-h)
    # P4 = d*(g-e)
    # P5 = (a+d)*(e+h)
    # P6 = (b-d)*(g+h)
    # P7 = (c-a)*(f+g)
    #
    # C = [[P1+P4-P5+P7, P3+P5],
    #      [P2+P5,        P1-P2+P3+P6]]

    # For 2x2x2, the rank-7 decomposition:
    # U factors (for A indices a=i*2+j, j in 0,1; i in 0,1):
    #   a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # V factors (for B indices e=j*2+k, k in 0,1; j in 0,1):
    #   e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    # W factors (for C indices c=k*2+i, i in 0,1; k in 0,1):
    #   C[0,0]=(0,0), C[0,1]=(0,1), C[1,0]=(1,0), C[1,1]=(1,1)

    # The 7 rank-1 terms:
    # Term 1: (a+b)*(e+f) -> contributes to C[0,0] and C[1,1]
    # Term 2: (c+d)*e     -> contributes to C[1,0]
    # Term 3: a*(f-h)     -> contributes to C[0,1] and C[1,1]
    # Term 4: d*(g-e)     -> contributes to C[0,0]
    # Term 5: (a+d)*(e+h) -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # Term 6: (b-d)*(g+h) -> contributes to C[1,1]
    # Term 7: (c-a)*(f+g) -> contributes to C[0,0] and C[1,0]

    # Let me define the 7 factors more carefully:
    # For term t, we have U[:,t] * V[:,t] * W[:,t]
    #
    # Term 1: U=[1,1,0,0] (a,b), V=[1,1,1,1] (e,f,g,h), W=[1,0,0,1] (C00,C01,C10,C11)
    #   Check: a*e + a*f + b*e + b*f -> this is (a+b)(e+f)
    #   But we need this to contribute P1 to C[0,0] and P1 to C[1,1]
    #   Actually C[0,0] gets P1, C[1,1] gets P1
    #   So W = [1, 0, 0, 1] means: C[0,0] += P1, C[1,1] += P1. Correct.
    #
    # Term 2: U=[0,0,1,1] (c,d), V=[1,0,0,0] (e), W=[0,0,1,0] (C10)
    #   c*e + d*e = (c+d)*e = P2. Contributes to C[1,0]. Correct.
    #
    # Term 3: U=[1,0,0,0] (a), V=[0,1,-1,0] (f,-h), W=[0,1,0,1] (C01,C11)
    #   a*f - a*h = a*(f-h) = P3. Contributes to C[0,1] and C[1,1]. 
    #   But in C formula: C[0,1] = P3+P5, C[1,1] = P1-P2+P3+P6
    #   So P3 contributes +1 to C[0,1] and +1 to C[1,1]. Correct.
    #
    # Term 4: U=[0,0,0,1] (d), V=[0,0,1,-1] (g,-e), W=[1,0,0,0] (C00)
    #   d*g - d*e = d*(g-e) = P4. Contributes to C[0,0]. Correct.
    #
    # Term 5: U=[1,0,0,1] (a,d), V=[1,0,0,1] (e,h), W=[1,1,1,1] (all)
    #   a*e + a*h + d*e + d*h = (a+d)(e+h) = P5. Contributes +1 to all of C.
    #   C[0,0] = P1+P4-P5+P7 -> P5 contributes -1. So W should be [-1,-1,-1,-1]
    #   Wait, let me re-check. C[0,0] = P1+P4-P5+P7, C[0,1] = P3+P5
    #   C[1,0] = P2+P5, C[1,1] = P1-P2+P3+P6
    #   So P5 contributes: C[0,0]: -1, C[0,1]: +1, C[1,0]: +1, C[1,1]: 0
    #   Hmm, that doesn't match W=[1,1,1,1]. Let me redo.
    #
    # Actually, I think I need to be more careful. Let me just verify numerically.

    def build_strassen_2x2x4():
        """Build rank-7 decomposition for 2x2 x 2x2 multiplication, then extend to 2x2 x 2x4."""
        # For 2x2 x 2x2, rank 7
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # A indices: 0=a, 1=b, 2=c, 3=d
        # B indices: 0=e, 1=f, 2=g, 3=h
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11

        U7 = np.zeros((4, 7), dtype=np.float64)
        V7 = np.zeros((4, 7), dtype=np.float64)
        W7 = np.zeros((4, 7), dtype=np.float64)

        # P1 = (a+b)*(e+f), contributes to C00 (+1) and C11 (+1)
        U7[0, 0] = 1; U7[1, 0] = 1
        V7[0, 0] = 1; V7[1, 0] = 1
        W7[0, 0] = 1; W7[3, 0] = 1

        # P2 = (c+d)*e, contributes to C10 (+1)
        U7[2, 1] = 1; U7[3, 1] = 1
        V7[0, 1] = 1
        W7[2, 1] = 1

        # P3 = a*(f-h), contributes to C01 (+1) and C11 (+1)
        U7[0, 2] = 1
        V7[1, 2] = 1; V7[3, 2] = -1
        W7[1, 2] = 1; W7[3, 2] = 1

        # P4 = d*(g-e), contributes to C00 (+1)
        U7[3, 3] = 1
        V7[2, 3] = 1; V7[0, 3] = -1
        W7[0, 3] = 1

        # P5 = (a+d)*(e+h), contributes to C00 (-1), C01 (+1), C10 (+1)
        U7[0, 4] = 1; U7[3, 4] = 1
        V7[0, 4] = 1; V7[3, 4] = 1
        W7[0, 4] = -1; W7[1, 4] = 1; W7[2, 4] = 1

        # P6 = (b-d)*(g+h), contributes to C11 (+1)
        U7[1, 5] = 1; U7[3, 5] = -1
        V7[2, 5] = 1; V7[3, 5] = 1
        W7[3, 5] = 1

        # P7 = (c-a)*(f+g), contributes to C00 (+1) and C10 (+1)
        U7[2, 6] = 1; U7[0, 6] = -1
        V7[1, 6] = 1; V7[2, 6] = 1
        W7[0, 6] = 1; W7[2, 6] = 1

        return U7, V7, W7

    def verify_decomposition(U, V, W, T, tol=1e-6):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    # Now for 4x4 x 4x8:
    # Split into blocks. A is 4x4, B is 4x8, C is 8x4.
    # We can view this as 4 blocks of 2x2 x 2x4 in the A dimension,
    # or better: 2 blocks of 2x4 x 4x4 (top and bottom halves of A).
    #
    # Actually, the cleanest approach:
    # A (4x4) split into four 2x2 blocks: A11, A12, A21, A22
    # B (4x8) split into two 2x4 blocks (top: rows 0-1, bottom: rows 2-3) 
    #   but each is 2x4, so B_top = B[0:2, :], B_bot = B[2:4, :]
    # C (8x4) = C_top (4x4) and C_bot (4x4)
    #
    # C_top = A11 * B_top[0:2,:] + A12 * B_bot[0:2,:] ... this gets complicated.
    #
    # Simpler: think of it as two independent 4x4x4 problems.
    # Left: A (4x4) * B_left (4x4, columns 0-3) = C_left (4x4, rows 0-3 of C)
    # Right: A (4x4) * B_right (4x4, columns 4-7) = C_right (4x4, rows 4-7 of C)
    #
    # For each 4x4x4, use rank 49 decomposition (Strassen-based).
    # Total rank = 2 * 49 = 98.

    # Build rank-49 decomposition for 4x4x4:
    # Use 2x2 blocks with Strassen's rank-7.
    # A (4x4) -> 4 blocks of 2x2: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B (4x4) -> 4 blocks of 2x2
    # C (4x4) -> 4 blocks of 2x2
    #
    # Each C_block = sum over 2 A_blocks * 2 B_blocks = 4 terms
    # Each 2x2 x 2x2 multiplication has rank 7
    # So each C_block needs 4*7 = 28 rank-1 terms
    # Total for 4 C_blocks: 4 * 28 = 112... that's too many.
    #
    # Actually, Strassen's algorithm for 4x4x4 doesn't simply multiply.
    # The standard approach: recursively apply Strassen.
    # For 4x4x4: split into 2x2 blocks, use 7 recursive calls of 2x2x2 (rank 7 each)
    # But the inputs to each recursive call are sums/differences of 2x2 blocks.
    #
    # The rank of 4x4x4 using Strassen recursion:
    # r(4) = 7 * r(2) = 7 * 7 = 49
    #
    # Let me build this explicitly.
    #
    # For 4x4 x 4x4:
    # A = [[A11, A12], [A21, A22]] where each Ai_j is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    #
    # C11 = P1 - P4 + P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P7
    # C22 = P1 - P2 + P3 + P6
    #
    # Each Pi is a 2x2 x 2x2 product, which has rank 7.
    # So each Pi contributes 7 rank-1 terms.
    # Total: 7 * 7 = 49 rank-1 terms.
    #
    # But wait - the inputs to the 2x2x2 multiplications are SUMS of 2x2 blocks.
    # So U for Pi involves combinations of A-blocks.
    #
    # Let me build this step by step.
    #
    # A-blocks (each 2x2, linearized):
    # A11: indices 0,1,2,3  (rows 0-1, cols 0-1)
    # A12: indices 4,5,6,7  (rows 0-1, cols 2-3)
    # A21: indices 8,9,10,11  (rows 2-3, cols 0-1)
    # A22: indices 12,13,14,15  (rows 2-3, cols 2-3)
    #
    # B-blocks:
    # B11: indices 0,1,2,3
    # B12: indices 4,5,6,7
    # B21: indices 8,9,10,11
    # B22: indices 12,13,14,15
    #
    # C-blocks:
    # C11: indices 0,1,2,3
    # C12: indices 4,5,6,7
    # C21: indices 8,9,10,11
    # C22: indices 12,13,14,15

    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4 x 4x4 matrix multiplication."""
        U7, V7, W7 = build_strassen_2x2x4()  # rank-7 for 2x2x2

        # For each of the 7 Strassen products Pi, we need a 2x2x2 multiplication
        # The input to each is a linear combination of A-blocks, B-blocks
        #
        # Define the linear combinations for each Pi:
        # P1 = (A11+A22) * (B11+B22)
        # P2 = (A21+A22) * B11
        # P3 = A11 * (B12-B22)
        # P4 = A22 * (B21-B11)
        # P5 = (A11+A12) * B22
        # P6 = (A21-A11) * (B11+B12)
        # P7 = (A12-A22) * (B21+B22)
        #
        # For U (A side), each Pi uses a specific combination of A-blocks:
        # P1: A11 + A22
        # P2: A21 + A22
        # P3: A11
        # P4: A22
        # P5: A11 + A12
        # P6: A21 - A11
        # P7: A12 - A22
        #
        # For V (B side):
        # P1: B11 + B22
        # P2: B11
        # P3: B12 - B22
        # P4: B21 - B11
        # P5: B22
        # P6: B11 + B12
        # P7: B21 + B22
        #
        # For W (C side), each C-block gets contributions from various Pi:
        # C11 = P1 - P4 + P5 - P7
        # C12 = P3 + P5
        # C21 = P2 + P7
        # C22 = P1 - P2 + P3 + P6

        # A-block index ranges (linear):
        # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
        # B-block index ranges:
        # B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
        # C-block index ranges:
        # C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15

        # U for 4x4x4: (16, 49)
        # V for 4x4x4: (16, 49)
        # W for 4x4x4: (16, 49)

        U4 = np.zeros((16, 49), dtype=np.float64)
        V4 = np.zeros((16, 49), dtype=np.float64)
        W4 = np.zeros((16, 49), dtype=np.float64)

        # For each Strassen product Pi (i = 0..6), add 7 rank-1 terms
        # The 7 terms come from the rank-7 2x2x2 decomposition (U7, V7, W7)
        #
        # U7 has shape (4, 7): each row corresponds to an element of the 2x2 input matrix
        # V7 has shape (4, 7)
        # W7 has shape (4, 7)
        #
        # For Pi, the A-input is a linear combination of A-blocks.
        # U7[r, t] gives the coefficient for element r of the 2x2 A-input in term t.
        # The 2x2 A-input's elements map to specific positions in the 4x4 A matrix.

        # Define which A-blocks each Pi uses and with what signs:
        # Pi -> list of (block_index, sign) where block_index: 0=A11, 1=A12, 2=A21, 3=A22
        A_combos = [
            [(0, 1), (3, 1)],   # P1: A11 + A22
            [(2, 1), (3, 1)],   # P2: A21 + A22
            [(0, 1)],           # P3: A11
            [(3, 1)],           # P4: A22
            [(0, 1), (1, 1)],   # P5: A11 + A12
            [(2, 1), (0, -1)],  # P6: A21 - A11
            [(1, 1), (3, -1)],  # P7: A12 - A22
        ]

        B_combos = [
            [(0, 1), (3, 1)],   # P1: B11 + B22
            [(0, 1)],           # P2: B11
            [(1, 1), (3, -1)],  # P3: B12 - B22
            [(2, 1), (0, -1)],  # P4: B21 - B11
            [(3, 1)],           # P5: B22
            [(0, 1), (1, 1)],   # P6: B11 + B12
            [(2, 1), (3, 1)],   # P7: B21 + B22
        ]

        # C-block contributions: Cj = sum_i coeff[i] * Pi
        # C11 = P1 - P4 + P5 - P7
        # C12 = P3 + P5
        # C21 = P2 + P7
        # C22 = P1 - P2 + P3 + P6
        C_coeffs = [
            [1, 0, 0, -1, 1, 0, -1],   # C11: P1 - P4 + P5 - P7
            [0, 0, 1, 0, 1, 0, 0],     # C12: P3 + P5
            [0, 1, 0, 0, 0, 0, 1],     # C21: P2 + P7
            [1, -1, 1, 0, 0, 1, 0],    # C22: P1 - P2 + P3 + P6
        ]

        # Block index to linear offset:
        # Block 0 (11): offset 0, Block 1 (12): offset 4, Block 2 (21): offset 8, Block 3 (22): offset 12
        block_offsets = [0, 4, 8, 12]

        for pi in range(7):
            # For each of the 7 rank-1 terms in the 2x2x2 decomposition
            for t in range(7):
                col = pi * 7 + t

                # U: combine A-blocks
                for blk, sign in A_combos[pi]:
                    offset = block_offsets[blk]
                    for r in range(4):
                        U4[offset + r, col] += sign * U7[r, t]

                # V: combine B-blocks
                for blk, sign in B_combos[pi]:
                    offset = block_offsets[blk]
                    for r in range(4):
                        V4[offset + r, col] += sign * V7[r, t]

                # W: each C-block gets its coefficient
                for cj in range(4):
                    coeff = C_coeffs[cj][pi]
                    if coeff != 0:
                        offset = block_offsets[cj]
                        for r in range(4):
                            W4[offset + r, col] += coeff * W7[r, t]

        return U4, V4, W4

    # Build the 4x4x4 decomposition
    U4, V4, W4 = build_4x4x4_strassen()

    # Verify the 4x4x4 decomposition
    T4 = np.zeros((16, 16, 16), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                T4[a, b, c] = 1

    if not verify_decomposition(U4, V4, W4, T4, tol=1e-6):
        # Fallback: schoolbook for 4x4x4
        U4 = np.zeros((16, 64), dtype=np.int64)
        V4 = np.zeros((16, 64), dtype=np.int64)
        W4 = np.zeros((16, 64), dtype=np.int64)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i * 4 + j, col] = 1
                    V4[j * 4 + k, col] = 1
                    W4[k * 4 + i, col] = 1
                    col += 1

    rank4 = U4.shape[1]  # should be 49 if Strassen works, else 64

    # Now extend to 4x4 x 4x8:
    # Left block: A (4x4) * B[0:4, 0:4] = C[0:4, :]  (rows 0-3 of C, which is 4x4)
    # Right block: A (4x4) * B[0:4, 4:8] = C[4:8, :]  (rows 4-7 of C, which is 4x4)
    #
    # U: (16, 2*rank4) - same A for both blocks
    # V: (32, 2*rank4) - B indices: j*8+k for left (k=0..3), j*8+k for right (k=4..7)
    # W: (32, 2*rank4) - C indices: k*4+i for left (k=0..3), k*4+i for right (k=4..7)

    total_rank = 2 * rank4
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((32, total_rank), dtype=np.float64)
    W = np.zeros((32, total_rank), dtype=np.float64)

    # Left block (k=0..3)
    for col in range(rank4):
        U[:, col] = U4[:, col]
        # V: B index (j, k) for k in 0..3 -> linear index j*8+k
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, col] = V4[j * 4 + k, col]
        # W: C index (k, i) for k in 0..3 -> linear index k*4+i
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, col] = W4[k * 4 + i, col]

    # Right block (k=4..7)
    for col in range(rank4):
        col2 = rank4 + col
        U[:, col2] = U4[:, col]
        # V: B index (j, k) for k in 4..7 -> linear index j*8+k, where k' = k-4
        for j in range(4):
            for k in range(4):
                V[j * 8 + (k + 4), col2] = V4[j * 4 + k, col]
        # W: C index (k, i) for k in 4..7 -> linear index k*4+i, where k' = k-4
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i, col2] = W4[k * 4 + i, col]

    # Verify the full 4x4x8 decomposition
    if not verify_decomposition(U, V, W, T, tol=1e-6):
        # Fallback to schoolbook
        U = U_sb.astype(np.float64)
        V = V_sb.astype(np.float64)
        W = W_sb.astype(np.float64)
    else:
        # Check if entries are half-integers (should be, since all coefficients are integers)
        pass

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
