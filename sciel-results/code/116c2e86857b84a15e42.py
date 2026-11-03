"""Block-Strassen decomposition for 2x4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _mat44_mul(U, V):
    """Multiply 4x4 U by 4x4 V (schoolbook), returns 4x4."""
    W = [[0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            s = 0
            for t in range(4):
                s += U[r][t] * V[t][c]
            W[r][c] = s
    return W

def _mat44_add(A, B):
    """Add two 4x4 matrices element-wise."""
    return [[A[r][c] + B[r][c] for c in range(4)] for r in range(4)]

def _mat44_sub(A, B):
    """Subtract B from A element-wise."""
    return [[A[r][c] - B[r][c] for c in range(4)] for r in range(4)]

def _block_decompose(M, row_start, col_start):
    """Extract 4 4x4 blocks from matrix M starting at (row_start, col_start)."""
    tl = [[M[row_start + r][col_start + c] for c in range(4)] for r in range(4)]
    tr = [[M[row_start + r][col_start + 4 + c] for c in range(4)] for r in range(4)]
    bl = [[M[row_start + 4 + r][col_start + c] for c in range(4)] for r in range(4)]
    br = [[M[row_start + 4 + r][col_start + 4 + c] for c in range(4)] for r in range(4)]
    return tl, tr, bl, br

def _assemble_blocks(tl, tr, bl, br, row_start, col_start):
    """Assemble 4 4x4 blocks into a matrix at position (row_start, col_start)."""
    M = [[0] * 8 for _ in range(8)]
    for r in range(4):
        for c in range(4):
            M[row_start + r][col_start + c] = tl[r][c]
            M[row_start + r][col_start + 4 + c] = tr[r][c]
            M[row_start + 4 + r][col_start + c] = bl[r][c]
            M[row_start + 4 + r][col_start + 4 + c] = br[r][c]
    return M

def construct():
    """Block-Strassen: outer 2x2 blocks of 2x4x4 tensor, each 4x4x4 via schoolbook.
    Rank = 7 (outer Strassen) * 16 (inner schoolbook) = 112."""
    # Outer Strassen: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # 7 products: M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22),
    #             M4=A22(B21-B11), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12),
    #             M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6

    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 112

    U = [[0] * rank for _ in range(n * m)]   # 8 x 112
    V = [[0] * rank for _ in range(m * p)]   # 16 x 112
    W = [[0] * rank for _ in range(p * n)]   # 8 x 112

    # Define the 7 outer Strassen terms: (U_factor, V_factor, W_contributions)
    # U_factor: 2x2 matrix over 4x4 blocks (indices 0..7 for A)
    # V_factor: 4x4 matrix over 4x4 blocks (indices 0..15 for B)
    # W_contributions: list of (C_block_index, sign) where C_block_index 0..3
    #   0=C11, 1=C12, 2=C21, 3=C22

    # A block indices: A11->0, A12->1, A21->2, A22->3
    # B block indices: B11->0, B12->1, B21->2, B22->3
    # C block indices: C11->0, C12->1, C21->2, C22->3

    # Each outer term has a U pattern (which A blocks, with signs) and V pattern (which B blocks, with signs)
    # M1: U=(A11+A22), V=(B11+B22)
    # M2: U=(A21+A22), V=B11
    # M3: U=A11, V=(B12-B22)
    # M4: U=A22, V=(B21-B11)
    # M5: U=(A11+A12), V=B22
    # M6: U=(A21-A11), V=(B11+B12)
    # M7: U=(A12-A22), V=(B21+B22)

    # For each outer term t, the inner product is a 4x4x4 multiplication.
    # The U factor for the inner: for each of the 4 rows of the 4x4 U-submatrix,
    #   it's a linear combination of the A-block rows.
    # Similarly for V.

    # Let's build this systematically.
    # A is 2x4 (n=2 rows of 4 cols each -> 2x8 total, but as 2x2 blocks of 4x4)
    # Wait, A is 2x4, so it's 2 rows, 4 cols. As blocks: each block is 2x4? No.
    # A is 2x4: 2 rows, 4 columns. Split into 2x2 blocks of size 1x2? That doesn't work well.
    # 
    # Actually, A is 2x4. The "outer" 2x2 block structure means:
    #   A11 = A[0:1, 0:2], A12 = A[0:1, 2:4], A21 = A[1:2, 0:2], A22 = A[1:2, 2:4]
    # Each is 1x2. But then the inner multiplication is 1x2 times 2x2... that's not 4x4x4.
    #
    # Let me reconsider. The 2x4x4 tensor: A is 2x4, B is 4x4, C is 2x4.
    # Split A into 2x2 blocks: A11(1x2), A12(1x2), A21(1x2), A22(1x2)
    # Split B into 2x2 blocks: B11(2x2), B12(2x2), B21(2x2), B22(2x2)
    # Split C into 2x2 blocks: C11(1x2), C12(1x2), C21(1x2), C22(1x2)
    #
    # Strassen gives 7 products of (1x2)*(2x2) = 1x2 matrices.
    # Each such product has rank 1*2*2 = 4 (schoolbook).
    # Total rank = 7 * 4 = 28. That matches the known result!
    #
    # But wait, that's the same as the Strassen-on-2x2x2 approach giving 28.
    # Let me think about whether we can do better.
    #
    # Alternative: Split A into blocks of size 2x2 (but A is only 2 rows, so each block is 2x2? No, A is 2x4, so 2 blocks of 2x2 side by side: A = [A1 | A2] where A1, A2 are 2x2).
    # B is 4x4: split into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C is 2x4: C = [C1 | C2] where C1, C2 are 2x2.
    #
    # C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    # This is like a 1x2 block structure on the left. Not directly Strassen-able.
    #
    # The best we can do with Strassen on the 2x2x2 structure (treating the 4s as scalars)
    # gives rank 28. To beat that, we need a different approach.
    #
    # Let me try: use the fact that the tensor is 2x4x4 and find a direct low-rank decomposition.
    # The schoolbook rank is 32. Strassen gives 28. Can we do better?
    #
    # Let me try a hybrid: use Strassen on the 2x2x2 block structure (giving 7 terms),
    # but for each term, the "scalar" multiplication is actually a 1x2 * 2x2 product.
    # Each such product: (1x2) * (2x2) -> (1x2). Rank of this tensor = 4.
    # Total: 7 * 4 = 28.
    #
    # Alternatively, can we share some computations across the 7 terms?
    # The 7 terms involve specific linear combinations of A-blocks and B-blocks.
    # If we can find a rank-3 decomposition of the 1x2x2 tensor (instead of rank 4),
    # total would be 7*3 = 21. But rank of M_1,2,2 is 4 (it's just matrix mult 1x2*2x2).
    #
    # So 28 seems to be the Strassen limit. Let me just implement it cleanly.

    # Reset for the 2x2x2 Strassen approach with 1x2 x 2x2 inner products.
    # A: 2x4, B: 4x4, C: 2x4
    # Blocks: A11=A[0,0:2], A12=A[0,2:4], A21=A[1,0:2], A22=A[1,2:4] (each 1x2)
    #         B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4] (each 2x2)
    #         C11=C[0,0:2], C12=C[0,2:4], C21=C[1,0:2], C22=C[1,2:4] (each 1x2)

    rank = 28
    U = [[0] * rank for _ in range(8)]   # n*m = 2*4 = 8
    V = [[0] * rank for _ in range(16)]  # m*p = 4*4 = 16
    W = [[0] * rank for _ in range(8)]   # p*n = 4*2 = 8

    # 7 Strassen terms, each with rank 4 (1x2 * 2x2 schoolbook)
    # Term definitions: (U_blocks, V_blocks, C_blocks)
    # U_blocks: list of (A_block_idx, coeff)
    # V_blocks: list of (B_block_idx, coeff)
    # C_blocks: list of (C_block_idx, coeff)

    # A block indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B block indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C block indices: 0=C11, 1=C12, 2=C21, 3=C22

    terms = [
        # M1 = (A11+A22)(B11+B22) -> C11+=M1, C22+=M1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # M2 = (A21+A22)B11 -> C21+=M2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'C': [(2, 1)]},
        # M3 = A11(B12-B22) -> C12+=M3, C22+=M3
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # M4 = A22(B21-B11) -> C11+=M4, C21+=M4
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        # M5 = (A11+A12)B22 -> C11-=M5, C12+=M5
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # M6 = (A21-A11)(B11+B12) -> C22+=M6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # M7 = (A12-A22)(B21+B22) -> C11+=M7
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]

    # A block row/col ranges: A11=(row0,col0:2), A12=(row0,col2:4), A21=(row1,col0:2), A22=(row1,col2:4)
    # A block (idx) -> (row_offset, col_offset) in the 2x4 A matrix
    A_block_pos = {0: (0, 0), 1: (0, 2), 2: (1, 0), 3: (1, 2)}  # 1x2 blocks
    # B block (idx) -> (row_offset, col_offset) in the 4x4 B matrix
    B_block_pos = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}  # 2x2 blocks
    # C block (idx) -> (row_offset, col_offset) in the 2x4 C matrix
    C_block_pos = {0: (0, 0), 1: (0, 2), 2: (1, 0), 3: (1, 2)}  # 1x2 blocks

    col = 0
    for term in terms:
        # For each of the 4 inner scalar products (rank 4 for 1x2*2x2):
        # The 1x2 U-block has 2 columns, the 2x2 V-block has 2 rows and 2 cols.
        # Inner product: for each of the 2 output columns c_out:
        #   sum over t in {0,1} of U[t] * V[t, c_out]
        # So 2 output columns, each with 2 terms = 4 scalar products.
        for c_out in range(2):
            for t in range(2):
                # U factor: for each A-block in term['U'], for each col in the 1x2 block
                #   U[row_offset, col_offset + t] gets the coefficient
                for a_idx, a_coeff in term['U']:
                    a_row, a_col = A_block_pos[a_idx]
                    U[a_row * 4 + a_col + t][col] = a_coeff
                # V factor: for each B-block in term['V'], for each row t and col c_out
                #   V[(row_offset + t)*4 + col_offset + c_out] gets the coefficient
                for b_idx, b_coeff in term['V']:
                    b_row, b_col = B_block_pos[b_idx]
                    V[(b_row + t) * 4 + b_col + c_out][col] = b_coeff
                # W factor: for each C-block in term['C']
                #   C[row_offset, col_offset + c_out] gets the coefficient
                for c_idx, c_coeff in term['C']:
                    c_row, c_col = C_block_pos[c_idx]
                    W[(c_col + c_out) * 2 + c_row][col] = c_coeff
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
