"""Matrix multiplication 2,2,9 – rank-32 exact tensor decomposition (Strassen for 8 cols, schoolbook for 1)."""

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    # n=2, m=2, p=9
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (18, rank)
    #
    # We use Strassen (7 multiplications) for columns 0..6 of B (7 columns),
    # and schoolbook (4 multiplications) for column 8 of B.
    # Total rank = 7 + 4 = 11.
    #
    # Wait, that's not right. The tensor rank for matrix multiplication is about
    # the number of scalar multiplications. For a 2x2 * 2x9 matrix, we have
    # 2*2*9 = 36 scalar multiplications in schoolbook.
    #
    # Strassen reduces 2x2*2x2 from 8 to 7. So for 2x2*2x9, we can view it as
    # 9 independent 2x2*2x1 multiplications, each needing 2 multiplications = 18.
    # That's still schoolbook per column.
    #
    # Alternatively, view as 4 independent 2x2*2x2 blocks (using 7 columns) 
    # plus one 2x2*2x1 (using 1 column). 
    # 4 blocks * 7 + 1 block * 2 = 28 + 2 = 30? No, that's not right either.
    #
    # Let me think more carefully. The matrix multiplication tensor for (2,2,9)
    # is the product of a (2,2) matrix and a (2,9) matrix, giving a (2,9) result.
    # 
    # We can partition the 9 columns of B into:
    # - 7 columns: use Strassen to multiply 2x2 by 2x7 (treat as 2x2 * [2x2, 2x2, 2x2, 2x1])
    #   Actually, Strassen works on 2x2 * 2x2. For 2x2 * 2x7, we can do:
    #   - Split 2x7 into [2x2, 2x2, 2x2, 2x1]
    #   - Use Strassen for the three 2x2 blocks: 3 * 7 = 21 mults
    #   - Use schoolbook for the 2x1 block: 2 mults
    #   - Total: 23 mults for 2x2 * 2x7
    # - Plus 2 remaining columns (columns 7,8): use schoolbook for 2x2 * 2x2: 8 mults
    #   Wait, 7 + 2 = 9. So columns 0-6 (7 cols) and columns 7-8 (2 cols).
    #   2x2 * 2x2 for the last 2 columns: 8 mults (schoolbook) or 7 (Strassen)
    #
    # So total: 23 + 7 = 30? Or 23 + 8 = 31?
    #
    # Let me be more precise:
    # Columns 0,1,2,3,4,5,6 (7 columns): treat as 2x2 * 2x7
    #   = 2x2 * [B0 (2x2), B1 (2x2), B2 (2x2), B3 (2x1)]
    #   Strassen for each 2x2*2x2: 7 mults each, 3 blocks = 21
    #   Schoolbook for 2x2*2x1: 2 mults
    #   Subtotal: 23
    #
    # Columns 7,8 (2 columns): treat as 2x2 * 2x2
    #   Strassen: 7 mults
    #   Subtotal: 7
    #
    # Total rank: 23 + 7 = 30
    #
    # Let's verify this is correct by constructing the tensor.

    n, m, p = 2, 2, 9
    rank = 30

    U = [[0] * rank for _ in range(n * m)]   # 4 x 30
    V = [[0] * rank for _ in range(m * p)]   # 18 x 30
    W = [[0] * rank for _ in range(p * n)]   # 18 x 30

    col = 0

    # Helper: add a Strassen 2x2*2x2 block
    # A is 2x2 (indices in U: 0,1,2,3 for A[0,0],A[0,1],A[1,0],A[1,1])
    # B is 2x2 (indices in V: row*2+col for B[row,col])
    # C is 2x2 (indices in W: col*2+row for C[row,col])
    # Strassen 7 products:
    def add_strassen(b_row_offset, c_col_offset, rank_offset):
        # b_row_offset: row offset in B (0 or 1), so B indices are (b_row_offset*2 + 0, b_row_offset*2 + 1) for row 0, etc.
        # Actually B is 2x2, so B[0,0], B[0,1], B[1,0], B[1,1]
        # V index for B[r,c] = (b_row_offset + r) * p + (c_col_offset + c) ... no.
        # V has shape (m*p, rank) = (18, rank). V index for B[j,k] = j*p + k.
        # For a 2x2 sub-block of B starting at column c0:
        #   B[0,c0] -> V[0*p + c0] = V[c0]
        #   B[0,c0+1] -> V[c0+1]
        #   B[1,c0] -> V[1*p + c0] = V[p + c0]
        #   B[1,c0+1] -> V[p + c0 + 1]
        # W has shape (p*n, rank) = (18, rank). W index for C[i,k] = k*n + i.
        # For a 2x2 sub-block of C starting at column c0:
        #   C[0,c0] -> W[c0*2 + 0] = W[2*c0]
        #   C[0,c0+1] -> W[(c0+1)*2 + 0] = W[2*c0+2]
        #   C[1,c0] -> W[2*c0+1]
        #   C[1,c0+1] -> W[2*c0+3]

        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        b00 = 0 * p + c_col_offset
        b01 = 0 * p + (c_col_offset + 1)
        b10 = 1 * p + c_col_offset
        b11 = 1 * p + (c_col_offset + 1)

        c00 = (2 * c_col_offset) + 0
        c01 = (2 * (c_col_offset + 1)) + 0
        c10 = (2 * c_col_offset) + 1
        c11 = (2 * (c_col_offset + 1)) + 1

        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1  # A00
        U[3][col] += 1  # A11
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

        # M2 = (A10+A11)B00
        U[2][col] += 1  # A10
        U[3][col] += 1  # A11
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] += -1
        col += 1

        # M3 = A00(B01-B11)
        U[0][col] += 1  # A00
        V[b01][col] += 1
        V[b11][col] += -1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1

        # M4 = A11(B10-B00)
        U[3][col] += 1  # A11
        V[b10][col] += 1
        V[b00][col] += -1
        W[c00][col] += 1
        W[c10][col] += -1
        col += 1

        # M5 = (A00+A01)B11
        U[0][col] += 1  # A00
        U[1][col] += 1  # A01
        V[b11][col] += 1
        W[c00][col] += -1
        W[c01][col] += 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[2][col] += 1  # A10
        U[0][col] += -1  # A00
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1  # A01
        U[3][col] += -1  # A11
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c01][col] += 1
        col += 1

        return col

    # Helper: add a schoolbook 2x2*2x1 block
    def add_schoolbook_2x1(c_col_offset, rank_offset):
        # B is 2x1, C is 2x1
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # 4 products: A00*B0, A01*B1, A10*B0, A11*B1

        b0 = 0 * p + c_col_offset
        b1 = 1 * p + c_col_offset
        c0 = 2 * c_col_offset + 0
        c1 = 2 * c_col_offset + 1

        # A00*B0 -> C0
        U[0][col] += 1
        V[b0][col] += 1
        W[c0][col] += 1
        col += 1

        # A01*B1 -> C0
        U[1][col] += 1
        V[b1][col] += 1
        W[c0][col] += 1
        col += 1

        # A10*B0 -> C1
        U[2][col] += 1
        V[b0][col] += 1
        W[c1][col] += 1
        col += 1

        # A11*B1 -> C1
        U[3][col] += 1
        V[b1][col] += 1
        W[c1][col] += 1
        col += 1

        return col

    # Columns 0-6: 3 Strassen 2x2*2x2 blocks (cols 0-1, 2-3, 4-5) + 1 schoolbook 2x2*2x1 (col 6)
    col = add_strassen(0, 0, col)       # cols 0,1
    col = add_strassen(0, 2, col)       # cols 2,3
    col = add_strassen(0, 4, col)       # cols 4,5
    col = add_schoolbook_2x1(6, col)    # col 6

    # Columns 7-8: 1 Strassen 2x2*2x2 block
    col = add_strassen(0, 7, col)       # cols 7,8

    # Verify col == rank
    assert col == rank, f"col={col}, rank={rank}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

