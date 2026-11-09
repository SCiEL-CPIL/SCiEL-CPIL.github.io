"""Improved program for 3x4x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Use Strassen-style decomposition for 3x4 * 4x6
    # Split middle dimension 4 into 2+2
    # A (3x4) = [A1 (3x2), A2 (3x2)]
    # B (4x6) = [B1 (2x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    #
    # Pad A to 4x4: A_padded = [[A11, A12], [0, 0]] where A11, A12 are 2x2
    # Pad C to 4x6: C_padded = [[C11, C12], [0, 0]]
    #
    # Use Strassen on 4x4 * 4x6:
    # A_padded = [[A11, A12], [0, 0]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] (2x2 blocks of 2x3)
    #
    # Strassen multiplications (each is 2x2 * 2x3):
    # M1 = (A11+0) * (B11+B22) = A11 * (B11+B22)
    # M2 = (0+0) * B11 = 0
    # M3 = A11 * (B12-B22)
    # M4 = 0 * (B21-B11) = 0
    # M5 = (A11+A12) * B22
    # M6 = (0-A11) * (B11+B12) = -A11 * (B11+B12)
    # M7 = (A12-0) * (B21+B22) = A12 * (B21+B22)
    #
    # C11 (2x3) = M1 + M4 - M5 + M7 = A11*(B11+B22) - (A11+A12)*B22 + A12*(B21+B22)
    # C12 (2x3) = M3 + M5 = A11*(B12-B22) + (A11+A12)*B22
    # C21 (2x3) = M2 + M4 = 0 (this is the padded part, we don't need it)
    # C22 (2x3) = M1 - M2 + M3 + M6 = A11*(B11+B22) + A11*(B12-B22) - A11*(B11+B12)
    #
    # We only need C11 and C12 (the first 2 rows of C), since C is 3x6 and 
    # we padded to 4x6. Wait, no - we padded A from 3x4 to 4x4, so the output 
    # C_padded is 4x6, but we only need the first 3 rows.
    #
    # Actually, let me reconsider. The original A is 3x4, not 4x4.
    # Let me use a different approach.
    #
    # Direct Strassen-like for 3x4 * 4x6:
    # Split A (3x4) into A_top (2x4) and A_bot (1x4)
    # Split B (4x6) into B_left (4x3) and B_right (4x3)
    # C (3x6) = [C_left (3x3), C_right (3x3)]
    # C_left = A_top * B_left_top + A_bot * B_left_bot  (but this doesn't quite work)
    #
    # Let me try a cleaner approach:
    # A (3x4), B (4x6), C (3x6)
    # Split middle dim: A = [A1 (3x2), A2 (3x2)], B = [B1 (2x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    #
    # Now use the identity: X*Y + Z*W = (X+Z)*(Y+W) - X*W - Z*Y
    # So C = (A1+A2)*(B1+B2) - A1*B2 - A2*B1
    # This uses 3 products of size 3x2 * 2x6
    # Each 3x2 * 2x6 has schoolbook rank 3*2*6 = 36
    # Total rank = 3 * 36 = 108, which is worse!
    #
    # The Strassen trick works when you have 4 products and can reduce to 7.
    # Here we have 2 products, so the identity doesn't help.
    #
    # Let me try: pad A to 4x4 by adding a zero row.
    # A_padded (4x4) = [[A11, A12], [A21, A22]] where A21, A22 are 1x2 (not 2x2)
    # This doesn't give clean 2x2 blocks.
    #
    # Alternative: use the known result that 2x2x2 matrix multiplication has rank 7.
    # For 2x2 * 2x3, I believe the rank is 7 or 8.
    # Let me assume rank 8 for 2x2 * 2x3.
    #
    # If I can decompose 3x4 * 4x6 using 4 blocks of 2x2 * 2x3 (or similar),
    # and each takes rank 8, total might be less than 72.
    #
    # Actually, let me just return the schoolbook. Finding a better decomposition
    # for 3x4x6 is non-trivial and I'm not confident I can do it correctly.
    
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
