"""Matrix multiplication 4,4,8 decomposition.

Strategy: use the known rank-49 Strassen decomposition for 4x4x4 matrix
multiplication, and exploit the direct-sum structure of 4x4x8 to get rank-98.
The 4x4x8 tensor T_{i,j,k} = delta_{i,j,k} for C = A(4x4) * B(4x8) can be
decomposed as two independent 4x4x4 multiplications (left 4 cols and right 4 cols
of B).
"""
import time

DIMENSIONS = [4, 4, 8]

DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)
T0 = time.time()


def _check_deadline():
    return time.time() - T0 > DEADLINE


def construct():
    """Return {'U','V','W'} factor matrices for 4x4*4x8 matrix multiplication.

    Uses Strassen 4x4x4 (rank 49) composed as direct sum for two halves,
    giving total rank 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8

    # ------------------------------------------------------------------
    # Strassen 4x4x4 decomposition
    # ------------------------------------------------------------------
    # We use the standard Strassen scheme for 4x4 matrices partitioned into
    # four 2x2 blocks. Each 2x2x2 Strassen step has rank 7.
    #
    # A = [[A11, A12],
    #      [A21, A22]]   (each 2x2)
    # B = [[B11, B12],
    #      [B21, B22]]   (each 2x2)
    #
    # Strassen 2x2 products:
    #   P1 = (A11+A22)*(B11+B22)
    #   P2 = (A21+A22)*B11
    #   P3 = A11*(B12-B22)
    #   P4 = A22*(B21-B11)
    #   P5 = (A11+A12)*B22
    #   P6 = (A21-A11)*(B11+B12)
    #   P7 = (A12-A22)*(B21+B22)
    #
    # C11 = P1 - P4 + P6 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P2 + P5
    #
    # For 4x4x4: we have 4 such 2x2x2 blocks (top-left, top-right, bottom-left,
    # bottom-right of the 4x4 output), each using 7 scalar products => 28?
    # No wait - the Strassen recursion for 4x4 gives rank 7^2 = 49.
    #
    # Let me build it properly using the tensor decomposition.
    #
    # For 4x4x4, indices: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1,2,3}.
    # We partition i into (i0 in {0,1}, i1 in {0,1}), j into (j0, j1), k into (k0, k1).
    #
    # The 49 Strassen products correspond to all combinations of 7 Strassen
    # "directions" for the row-partition (i0, j0) and 7 for the col-partition
    # (j1, k1)? No, that's not right either.
    #
    # Actually, the standard approach: 4x4x4 = tensor product structure.
    # The rank-49 decomposition can be built as:
    #   U has 16 rows (4*4), V has 16 rows (4*4), W has 16 rows (4*4), rank 49.
    #
    # Let me use a cleaner construction. I'll define the Strassen 2x2x2
    # decomposition explicitly, then use it to build 4x4x4.
    #
    # 2x2x2 Strassen:
    #   U (4 rows x 7 cols): maps (a00,a01,a10,a11) -> linear forms
    #   V (4 rows x 7 cols): maps (b00,b01,b10,b11) -> linear forms
    #   W (4 rows x 7 cols): maps products -> (c00,c01,c10,c11)
    #
    # With flattening: A index = a00*1 + a01*2 + a10*4 + a11*8?
    # Let's use row-major: index(i,j) = i*2+j for 2x2.
    #
    # U rows (linear forms on A entries a00,a01,a10,a11):
    #   col0 (P1): a00 + a11  -> [1,0,0,1]
    #   col1 (P2): a10 + a11  -> [0,0,1,1]
    #   col2 (P3): a00        -> [1,0,0,0]
    #   col3 (P4): a11        -> [0,0,0,1]
    #   col4 (P5): a00 + a01  -> [1,1,0,0]
    #   col5 (P6): a10 - a00  -> [-1,0,1,0]
    #   col6 (P7): a01 - a11  -> [0,1,-1,0]  wait let me recheck.
    #
    # Standard Strassen 2x2:
    #   P1 = (A11+A22)(B11+B22)
    #   P2 = (A21+A22)B11
    #   P3 = A11(B12-B22)
    #   P4 = A22(B21-B11)
    #   P5 = (A11+A12)B22
    #   P6 = (A21-A11)(B11+B12)
    #   P7 = (A12-A22)(B21+B22)
    #
    # Using 0-based: A11=a[0,0], A12=a[0,1], A21=a[1,0], A22=a[1,1]
    # B11=b[0,0], B12=b[0,1], B21=b[1,0], B22=b[1,1]
    #
    # U (4x7), row index = i*2+j:
    #   P1: a00+a11 -> row0: [1,0,0,1]  (a00 at idx0, a01 at idx1, a10 at idx2, a11 at idx3)
    #   P2: a10+a11 -> row1? No, U is per-element.
    #
    # Let me just build U, V, W as full matrices.
    # U has shape (4, 7) where row r corresponds to A element at position r (0-3).
    # V has shape (4, 7) where row r corresponds to B element at position r.
    # W has shape (4, 7) where row r corresponds to C element at position r.
    #
    # C = A*B: c[i,k] = sum_j a[i,j]*b[j,k]
    # c[0,0] = a00*b00 + a01*b10
    # c[0,1] = a00*b01 + a01*b11
    # c[1,0] = a10*b00 + a11*b10
    # c[1,1] = a10*b01 + a11*b11

    # Strassen 2x2:
    # Products p[0..6]:
    # p0 = (a00+a11)*(b00+b11)
    # p1 = (a10+a11)*(b00)
    # p2 = (a00)*(b01-b11)
    # p3 = (a11)*(b10-b00)
    # p4 = (a00+a01)*(b11)
    # p5 = (a10-a00)*(b00+b01)
    # p6 = (a01-a11)*(b10+b11)
    #
    # c00 = p0 - p3 + p5 + p6
    # c01 = p2 + p4
    # c10 = p1 + p3
    # c11 = p0 - p2 + p1 + p4

    # U[4,7]: U[i,j] = coefficient of A[i] in product j
    # A flattened: a00=0, a01=1, a10=2, a11=3
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1],  # a11
    ]
    # V[4,7]: V[j,k] = coefficient of B[j] in product k
    # B flattened: b00=0, b01=1, b10=2, b11=3
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1],  # b11
    ]
    # W[4,7]: W[i,k] = coefficient of product p in C[i,k]
    # C flattened: c00=0, c01=1, c10=2, c11=3
    W2 = [
        [1, 0, 0, 0, 0, 1, 1],   # c00 = p0 - p3 + p5 + p6
        [0, 0, 1, 0, 1, 0, 0],   # c01 = p2 + p4
        [0, 1, 0, 1, 0, 0, 0],   # c10 = p1 + p3
        [1, 1, -1, 0, 1, 0, 0],  # c11 = p0 - p2 + p1 + p4
    ]

    # Build 4x4x4 Strassen (rank 49) from the 2x2x2 building block.
    # The 4x4 matrix multiplication tensor can be decomposed using the
    # "Strassen recursion": treat 4x4 as 2x2 blocks of 2x2, and apply
    # Strassen at the block level. But this gives rank 7 (block level) * 7 (within-block) = 49.
    #
    # More precisely: for 4x4x4, we have indices i,j,k in {0,1,2,3}.
    # Partition: i = (i_high, i_low) where i_high = i//2, i_low = i%2.
    # Similarly for j and k.
    #
    # The tensor T[i,j,k] = 1 if the multiplication gives that triple.
    #
    # Using the Kronecker product structure: the 4x4x4 multiplication tensor
    # is the "block" version. We can write:
    #   U_4[i,j, c] where c = (c_block, c_inner) with c_block in 0..6, c_inner in 0..6
    #
    # Actually, let me just directly construct it.
    #
    # For 4x4, A[i,j] with i,j in 0..3. The Strassen-4x4 decomposition:
    #
    # Level 1: partition A into 2x2 blocks:
    #   A = [[A00, A01], [A10, A11]] where each Axy is 2x2
    #   B = [[B00, B01], [B10, B11]]
    #
    # Block-level Strassen products (7 of them), each a 2x2 * 2x2 = 4x4 tensor:
    #   P_block[l] = U_block[l, block_idx_A] * V_block[l, block_idx_B]
    #
    # Then within each block product, we need to multiply two 2x2 matrices.
    # But wait - the block-level Strassen gives us linear combinations of
    # 2x2 blocks, and each "product" is a 2x2 x 2x2 multiplication.
    #
    # So the full 4x4x4 rank-49 decomposition:
    #   For each block-level Strassen product l (0..6) and inner Strassen product m (0..6):
    #     product = (sum over A-blocks U_block[l, ab] * A_block[ab]) * (sum over B-blocks V_block[l, bb] * B_block[bb])
    #
    #   where A_block and B_block are 2x2 submatrices.
    #
    #   Then within each such product, we use the 2x2 Strassen (rank 7).
    #
    #   Total rank = 7 * 7 = 49.
    #
    # Let me define the block-level Strassen (same as 2x2 Strassen but operating on blocks):
    #   Block U (4x7): U_b[bi, l] where bi = block index in A (0=A00, 1=A01, 2=A10, 3=A11)
    #   Block V (4x7): V_b[bb, l] where bb = block index in B
    #   Block W (4x7): W_b[ci, l] where ci = block index in C (0=C00, 1=C01, 2=C10, 3=C11)
    #
    # These are the same as U2, V2, W2 above (since it's the same 2x2 Strassen pattern).

    # Now, the full 4x4x4 decomposition:
    # U_4 has shape (16, 49): rows indexed by (i,j) in 0..15 (row-major 4x4)
    # V_4 has shape (16, 49): rows indexed by (j,k) in 0..15
    # W_4 has shape (16, 49): rows indexed by (k,i) in 0..15 (trace convention W[k*n+i])
    #
    # For column c = l*7 + m (l = block-level product 0..6, m = inner product 0..6):
    #
    #   U_4[i*4+j, c]:
    #     The A element a[i,j] belongs to block (i//2, j//2) with inner position (i%2, j%2).
    #     Block index in A: ab = (i//2)*2 + (j//2)
    #     Inner A index: ai = (i%2)*2 + (j%2)
    #     U_4[i*4+j, c] = U2[ab, l] * U2[ai, m]
    #
    #   V_4[j*4+k, c]:
    #     B element b[j,k] belongs to block (j//2, k//2) with inner (j%2, k%2).
    #     Block index in B: bb = (j//2)*2 + (k//2)
    #     Inner B index: bi = (j%2)*2 + (k%2)
    #     V_4[j*4+k, c] = V2[bb, l] * V2[bi, m]
    #
    #   W_4[k*4+i, c]:  (trace convention: W[k*n+i] = W[k*4+i])
    #     C element c[i,k] (C = A*B, so C[i,k]).
    #     Block index in C: cb = (i//2)*2 + (k//2)
    #     Inner C index: ci = (i%2)*2 + (k%2)
    #     W_4[k*4+i, c] = W2[cb, l] * W2[ci, m]

    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]

    for i in range(4):
        for j in range(4):
            ab = (i // 2) * 2 + (j // 2)
            ai = (i % 2) * 2 + (j % 2)
            for l in range(7):
                for m in range(7):
                    c = l * 7 + m
                    U4[i * 4 + j][c] = U2[ab][l] * U2[ai][m]

    for j in range(4):
        for k in range(4):
            bb = (j // 2) * 2 + (k // 2)
            bi = (j % 2) * 2 + (k % 2)
            for l in range(7):
                for m in range(7):
                    c = l * 7 + m
                    V4[j * 4 + k][c] = V2[bb][l] * V2[bi][m]

    for i in range(4):
        for k in range(4):
            cb = (i // 2) * 2 + (k // 2)
            ci = (i % 2) * 2 + (k % 2)
            for l in range(7):
                for m in range(7):
                    c = l * 7 + m
                    W4[k * 4 + i][c] = W2[cb][l] * W2[ci][m]

    if _check_deadline():
        # Fallback: return schoolbook (shouldn't happen, but safety)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ------------------------------------------------------------------
    # Build 4x4x8 as direct sum of two 4x4x4 multiplications
    # ------------------------------------------------------------------
    # A is 4x4, B is 4x8. Split B into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).
    # C = A * B = [A*B_left | A*B_right], both 4x4.
    #
    # The 4x4x8 tensor: T[i, j, k] where i=0..3 (row of A), j=0..3 (col of A / row of B),
    # k=0..7 (col of B / col of C).
    #
    # U: (n*m, rank) = (16, 98)
    #   For the left half (k=0..3): U[i*m+j, col] = U4[i*4+j, c] for c in 0..48
    #   For the right half (k=4..7): U[i*m+j, col] = U4[i*4+j, c] for c in 0..48
    #   (same U for both halves since A is the same)
    #
    # V: (m*p, rank) = (32, 98)
    #   V has rows indexed by (j, k) with j=0..3, k=0..7. Row index = j*8+k.
    #   For left half (k=0..3): V[j*8+k, col] = V4[j*4+k, c] for c in 0..48
    #   For right half (k=4..7): V[j*8+k, col] = V4[j*4+(k-4), c] for c in 0..48
    #
    # W: (p*n, rank) = (32, 98)
    #   W has rows indexed by (k, i) with k=0..7, i=0..3. Row index = k*4+i (trace convention W[k*n+i]).
    #   For left half (k=0..3): W[k*4+i, col] = W4[k*4+i, c] for c in 0..48
    #   For right half (k=4..7): W[k*4+i, col] = W4[(k-4)*4+i, c] for c in 0..48

    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # 16 x 98
    V = [[0] * rank for _ in range(m * p)]   # 32 x 98
    W = [[0] * rank for _ in range(p * n)]   # 32 x 98

    for i in range(4):
        for j in range(4):
            urow = i * 4 + j
            for c in range(49):
                # Left half
                U[urow][c] = U4[urow][c]
                # Right half
                U[urow][49 + c] = U4[urow][c]

    for j in range(4):
        for k in range(4):
            vrow_left = j * 8 + k
            vrow_right = j * 8 + (k + 4)
            v4row = j * 4 + k
            for c in range(49):
                V[vrow_left][c] = V4[v4row][c]
                V[vrow_right][49 + c] = V4[v4row][c]

    for i in range(4):
        for k in range(4):
            wrow_left = k * 4 + i
            wrow_right = (k + 4) * 4 + i
            w4row = k * 4 + i
            for c in range(49):
                W[wrow_left][c] = W4[w4row][c]
                W[wrow_right][49 + c] = W4[w4row][c]

    return {"U": U, "V": V, "W": W}
