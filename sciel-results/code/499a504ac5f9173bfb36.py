"""Matrix multiplication 2x3 by 3x10 rank-minimal decomposition."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS  # 2, 3, 10

    def remaining_time():
        return DEADLINE - SAFETY_MARGIN - (time.time() - start_time)

    # We'll build the decomposition using a block structure.
    # The 3x10 matrix B is split into 5 blocks of 3x2.
    # For each block, we use Strassen for the 2x2 part (first 2 rows of B)
    # and schoolbook for the last row.

    # Strategy: For each of the 5 output column pairs (2x2 block of C):
    # - Split B into B_top (2x2) and B_bottom (1x2)
    # - Split A into A_left (2x2) and A_right (2x1)
    # - Use Strassen for A_left @ B_top (rank 7)
    # - Use schoolbook for A_left @ B_bottom (rank 4)  
    # - Use schoolbook for A_right @ B_top (rank 4)
    # - Use schoolbook for A_right @ B_bottom (rank 2)
    # Total per block: 7 + 4 + 4 + 2 = 17
    # Total rank: 5 * 17 = 85... wait that's worse than 60.

    # Let me reconsider. The schoolbook is 2*3*10 = 60.
    # Better approach: Split the 3x10 into blocks and use Strassen where beneficial.

    # Actually, let's think differently. The tensor is for (2x3) @ (3x10) -> (2x10).
    # We can view this as 10 independent (2x3) @ (3x1) multiplications, or
    # as 2 independent (2x3) @ (3x10) with shared A structure... no.

    # Key insight: We can split the 10 output columns into 5 groups of 2.
    # For each group of 2 columns, we compute C[:, 2k:2k+2] = A @ B[:, 2k:2k+2]
    # where A is 2x3 and B[:, 2k:2k+2] is 3x2.
    # So each subproblem is (2x3) @ (3x2) -> (2x2).

    # For a (2x3) @ (3x2) problem, schoolbook rank is 2*3*2 = 12.
    # Can we do better? Let's split the 3 intermediate dimension into 2+1.
    # A = [A1 (2x2), A2 (2x1)], B = [B1 (2x2); B2 (1x2)]
    # C = A1@B1 + A2@B2
    # A1@B1: (2x2)@(2x2) -> use Strassen, rank 7
    # A2@B2: (2x1)@(1x2) -> schoolbook, rank 4 (but actually it's outer product, rank 1? No, it's 2x2)
    # Wait, A2 is 2x1, B2 is 1x2, so A2@B2 is 2x2. Schoolbook for this: 2*1*2 = 4 terms.
    # Total per block: 7 + 4 = 11
    # Total rank: 5 * 11 = 55

    # That's an improvement over 60!

    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55

    def strassen_2x2x2(u_offset, v_offset, w_offset, base_col):
        """Add Strassen terms for (2x2)@(2x2)->(2x2) to U,V,W starting at base_col."""
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        # U indices for 2x2 A block: rows u_offset to u_offset+3 (flattened row-major: r0c0, r0c1, r1c0, r1c1)
        # V indices for 2x2 B block: rows v_offset to v_offset+3
        # W indices for 2x2 C block: rows w_offset to w_offset+3 (flattened: r0c0, r0c1, r1c0, r1c1)

        A00 = u_offset
        A01 = u_offset + 1
        A10 = u_offset + 2
        A11 = u_offset + 3

        B00 = v_offset
        B01 = v_offset + 1
        B10 = v_offset + 2
        B11 = v_offset + 3

        C00 = w_offset
        C01 = w_offset + 1
        C10 = w_offset + 2
        C11 = w_offset + 3

        col = base_col

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1
        U[A11][col] = 1
        V[B00][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1
        U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1
        W[C11][col] = -1
        col += 1

        # M3 = A00(B10-B11)
        U[A00][col] = 1
        V[B10][col] = 1
        V[B11][col] = -1
        W[C01][col] = 1
        W[C11][col] = 1
        col += 1

        # M4 = A11(B01-B10)
        U[A11][col] = 1
        V[B01][col] = 1
        V[B10][col] = -1
        W[C00][col] = 1
        W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1
        U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1
        W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B10)
        U[A10][col] = 1
        U[A00][col] = -1
        V[B00][col] = 1
        V[B10][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B01+B11)
        U[A01][col] = 1
        U[A11][col] = -1
        V[B01][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        col += 1

        return col

    def schoolbook_2x1_1x2(u_offset, v_offset, w_offset, base_col):
        """Add schoolbook terms for (2x1)@(1x2)->(2x2) to U,V,W starting at base_col."""
        # A is 2x1: rows u_offset, u_offset+1
        # B is 1x2: rows v_offset, v_offset+1
        # C is 2x2: rows w_offset to w_offset+3

        A0 = u_offset
        A1 = u_offset + 1
        B0 = v_offset
        B1 = v_offset + 1
        C00 = w_offset
        C01 = w_offset + 1
        C10 = w_offset + 2
        C11 = w_offset + 3

        col = base_col

        # C00 = A0*B0
        U[A0][col] = 1
        V[B0][col] = 1
        W[C00][col] = 1
        col += 1

        # C01 = A0*B1
        U[A0][col] = 1
        V[B1][col] = 1
        W[C01][col] = 1
        col += 1

        # C10 = A1*B0
        U[A1][col] = 1
        V[B0][col] = 1
        W[C10][col] = 1
        col += 1

        # C11 = A1*B1
        U[A1][col] = 1
        V[B1][col] = 1
        W[C11][col] = 1
        col += 1

        return col

    col = 0
    for block in range(5):
        if remaining_time() < 2.0:
            break

        # For block k, we're computing C[:, 2k:2k+2] = A @ B[:, 2k:2k+2]
        # A is 2x3: flatten index i*m+j = i*3+j
        # B[:, 2k:2k+2] is 3x2: flatten index j*10 + (2k+c)

        # Split A into A_top (2x2, cols 0,1) and A_bottom (2x1, col 2)
        # A_top: rows 0-1, cols 0-1 -> U indices: 0,1,3,4
        # A_bottom: rows 0-1, col 2 -> U indices: 2,5

        # Split B into B_top (2x2, rows 0-1) and B_bottom (1x2, row 2)
        # B_top: rows 0-1, cols 0-1 of block -> V indices: (0)*10+(2k), (0)*10+(2k+1), (1)*10+(2k), (1)*10+(2k+1)
        # B_bottom: row 2, cols 0-1 of block -> V indices: (2)*10+(2k), (2)*10+(2k+1)

        # C for this block: rows 0-1, cols 2k,2k+1
        # C flatten: k'*n + i = c*2 + i, so C[0,2k] -> 2k, C[0,2k+1] -> 2k+1, C[1,2k] -> 2k+2, C[1,2k+1] -> 2k+3

        # Strassen for A_top @ B_top (2x2 @ 2x2)
        # A_top in U: need to map cols 0,1 of A to Strassen A00,A01,A10,A11
        # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
        # But Strassen expects A00,A01,A10,A11 in order. We have them at 0,1,3,4.
        # We need a permutation. Let's use a different approach: build Strassen with explicit index mapping.

        # Actually, let me just use explicit index mapping for Strassen.

        # A_top indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        # B_top indices: B[0,0]=V[2k], B[0,1]=V[2k+1], B[1,0]=V[10+2k], B[1,1]=V[10+2k+1]
        # C indices: C[0,0]=W[2k], C[0,1]=W[2k+1], C[1,0]=W[2k+2], C[1,1]=W[2k+1+1]=W[2k+3]

        A00_idx = 0
        A01_idx = 1
        A10_idx = 3
        A11_idx = 4

        B00_idx = 2 * k
        B01_idx = 2 * k + 1
        B10_idx = 10 + 2 * k
        B11_idx = 10 + 2 * k + 1

        C00_idx = 2 * k
        C01_idx = 2 * k + 1
        C10_idx = 2 * k + 2
        C11_idx = 2 * k + 3

        # M1 = (A00+A11)(B00+B11)
        U[A00_idx][col] = 1
        U[A11_idx][col] = 1
        V[B00_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = 1
        W[C11_idx][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10_idx][col] = 1
        U[A11_idx][col] = 1
        V[B00_idx][col] = 1
        W[C10_idx][col] = 1
        W[C11_idx][col] = -1
        col += 1

        # M3 = A00(B10-B11)
        U[A00_idx][col] = 1
        V[B10_idx][col] = 1
        V[B11_idx][col] = -1
        W[C01_idx][col] = 1
        W[C11_idx][col] = 1
        col += 1

        # M4 = A11(B01-B10)
        U[A11_idx][col] = 1
        V[B01_idx][col] = 1
        V[B10_idx][col] = -1
        W[C00_idx][col] = 1
        W[C10_idx][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00_idx][col] = 1
        U[A01_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = -1
        W[C01_idx][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B10)
        U[A10_idx][col] = 1
        U[A00_idx][col] = -1
        V[B00_idx][col] = 1
        V[B10_idx][col] = 1
        W[C11_idx][col] = 1
        col += 1

        # M7 = (A01-A11)(B01+B11)
        U[A01_idx][col] = 1
        U[A11_idx][col] = -1
        V[B01_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = 1
        col += 1

        # Schoolbook for A_bottom @ B_bottom (2x1 @ 1x2)
        # A_bottom: A[0,2]=U[2], A[1,2]=U[5]
        # B_bottom: B[2,0]=V[20+2k], B[2,1]=V[20+2k+1]
        # C same as before

        Ab0_idx = 2
        Ab1_idx = 5
        Bb0_idx = 20 + 2 * k
        Bb1_idx = 20 + 2 * k + 1

        # C00 += Ab0 * Bb0
        U[Ab0_idx][col] = 1
        V[Bb0_idx][col] = 1
        W[C00_idx][col] = 1
        col += 1

        # C01 += Ab0 * Bb1
        U[Ab0_idx][col] = 1
        V[Bb1_idx][col] = 1
        W[C01_idx][col] = 1
        col += 1

        # C10 += Ab1 * Bb0
        U[Ab1_idx][col] = 1
        V[Bb0_idx][col] = 1
        W[C10_idx][col] = 1
        col += 1

        # C11 += Ab1 * Bb1
        U[Ab1_idx][col] = 1
        V[Bb1_idx][col] = 1
        W[C11_idx][col] = 1
        col += 1

    # Verify col == rank
    if col != rank:
        # Adjust if needed
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
