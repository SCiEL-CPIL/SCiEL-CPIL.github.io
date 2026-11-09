"""Strassen-based rank-49 decomposition for 4x4 x 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    def urow(i, j):
        return i * m + j

    def vrow(j, k):
        return j * p + k

    def wrow(k, i):
        return k * n + i

    # For each top-level Strassen product s (0..6), the A-factor is a 2x2 matrix
    # formed from sums/differences of A-blocks, and B-factor is a 2x7 matrix
    # formed from sums/differences of B-blocks.
    #
    # We apply Strassen to the 2x2 x 2x7 product. The 2x7 B matrix is treated
    # as having 2 rows (each 1x7). Strassen for 2x2 x 2x7:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, ..., b06], [b10, ..., b16]]
    # C = A * B (2x7)
    #
    # Standard Strassen 7 products for 2x2 x 2x2, adapted to 2x2 x 2x7:
    # M1 = (a00+a11) * (b00+b10, b01+b11, ..., b06+b16)
    # M2 = (a10+a11) * (b00, b01, ..., b06)
    # M3 = a00 * (b10-b00, b11-b01, ..., b16-b06)
    # M4 = a11 * (b10-b00, b11-b01, ..., b16-b06)
    # M5 = (a00+a01) * (b10, b11, ..., b16)
    # M6 = (a10-a00) * (b00+b10, b01+b11, ..., b06+b16)
    # M7 = (a01-a11) * (b10+b00, b11+b01, ..., b16+b06)
    #
    # Wait, this gives 7 products but each is a scalar times a 1x7 vector = 7 elementary ops.
    # Total = 7 * 7 = 49 per top product, times 7 top products = 343. Too many.
    #
    # Actually, for 2x2 x 2x7, the rank is 7 (not 49). The 7 Strassen products
    # each produce a 1x7 vector, but we can combine them. Let me reconsider.
    #
    # Actually, I think the correct approach is:
    # 4x4 x 4x7: Top-level Strassen gives 7 products of 2x2 x 2x7.
    # Each 2x2 x 2x7 can be done with 7 scalar products (Strassen applied to the 2x2 part,
    # treating each column of the 2x7 as a 2-vector).
    # So total = 7 * 7 = 49.
    #
    # Let me verify: 2x2 x 2x7 multiplication.
    # C[0,:] = a00*B[0,:] + a01*B[1,:]
    # C[1,:] = a10*B[0,:] + a11*B[1,:]
    #
    # Strassen 7 products:
    # M1 = (a00+a11)*(B[0,:]+B[1,:])  -> 1x7
    # M2 = (a10+a11)*B[0,:]            -> 1x7
    # M3 = a00*(B[1,:]-B[0,:])         -> 1x7
    # M4 = a11*(B[1,:]-B[0,:])         -> 1x7
    # M5 = (a00+a01)*B[1,:]            -> 1x7
    # M6 = (a10-a00)*(B[0,:]+B[1,:])   -> 1x7
    # M7 = (a01-a11)*(B[0,:]+B[1,:])   -> 1x7
    #
    # C[0,:] = M1 + M3 - M5
    # C[1,:] = M1 + M2 - M4
    #
    # Wait, let me check:
    # C[0,0] = a00*B[0,0] + a01*B[1,0]
    # M1[0] + M3[0] - M5[0] = (a00+a11)(B[0,0]+B[1,0]) + a00(B[1,0]-B[0,0]) - (a00+a01)B[1,0]
    # = a00*B[0,0] + a00*B[1,0] + a11*B[0,0] + a11*B[1,0] + a00*B[1,0] - a00*B[0,0] - a00*B[1,0] - a01*B[1,0]
    # = a00*B[0,0] + a11*B[0,0] + a11*B[1,0] + a00*B[1,0] - a00*B[1,0] - a01*B[1,0]
    # = a00*B[0,0] + a11*(B[0,0]+B[1,0]) - a01*B[1,0]
    # That's not right. Let me use the standard Strassen formula.
    #
    # Standard Strassen for 2x2 x 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # For 2x2 x 2x7, B has 2 rows and 7 columns. B00 = B[0,0], B01 = B[0,1], ..., B06 = B[0,6]
    # B10 = B[1,0], B11 = B[1,1], ..., B16 = B[1,6]
    #
    # But Strassen requires B to be 2x2. So we apply Strassen to each 2x2 sub-block of B?
    # No, B is 2x7, not 2x2.
    #
    # I think the correct interpretation is that 2x2 x 2x7 has rank 7, and the 7 products
    # are scalar products (each involving one entry of A and one entry of B per column).
    #
    # Actually, let me just use the schoolbook for the bottom level and see if I can
    # get a valid decomposition, even if not minimal.
    #
    # Top-level Strassen for 4x4 x 4x7:
    # A is 4x4, B is 4x7.
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, :], B12 = B[2:4, :]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    #
    # M1 = (A11+A22)(B11+B12)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B11)
    # M4 = A22(B11-B12)  -- wait, standard is A22(B21-B11) but B only has 2 block rows
    #
    # Since B is 4x7, it has 2 block rows of 2x7 each. So B11 = B[0:2, :], B21 = B[2:4, :]
    # There's no B12 or B22 in the column direction because we're not splitting columns.
    #
    # So for 4x4 x 4x7:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11], [B21]] where each block is 2x7
    # C = [[C11, C12], [C21, C22]] where C11, C12 are 2x2, C21, C22 are 2x2
    #
    # Wait, C is 4x7, so C = [[C11, C12], [C21, C22]] where C11, C12, C21, C22 are 2x(7/2)?
    # No, 7 is not divisible by 2. So we can't split C into 2x2 blocks.
    #
    # I think the issue is that p=7 is odd, so we can't cleanly apply Strassen at the bottom level.
    #
    # Let me just use the schoolbook decomposition with rank 112, which is always correct.
    #
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
# EVOLVE-BLOCK-END
