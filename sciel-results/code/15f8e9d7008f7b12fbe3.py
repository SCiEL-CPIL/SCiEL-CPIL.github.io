"""Strassen-based tensor decomposition for 4x4 x 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We decompose the 4x4x8 multiplication into two 4x4x4 multiplications.
    # For each 4x4x4, we use the rank-49 Strassen decomposition (applying Strassen 2x2x2 recursively).
    # Total rank = 49 + 49 = 98.
    
    # First, define the base Strassen 2x2x2 decomposition.
    # For 2x2x2: A is 2x2, B is 2x2, C = A*B is 2x2.
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    # A indices: 0,1 (row 0), 2,3 (row 1) -> flat: a=0, b=1, c=2, d=3
    # B indices: 0,1 (row 0), 2,3 (row 1) -> flat: e=0, f=1, g=2, h=3
    # C indices: 0,1 (row 0), 2,3 (row 1) -> flat: c00=0, c01=1, c10=2, c11=3
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P0 = (a+d)(e+h)
    U2[0][0] += 1; U2[3][0] += 1
    V2[0][0] += 1; V2[3][0] += 1
    W2[0][0] += 1; W2[3][0] += 1
    # P1 = (c+d)e
    U2[2][1] += 1; U2[3][1] += 1
    V2[0][1] += 1
    W2[2][1] += 1; W2[3][1] -= 1
    # P2 = a(f-h)
    U2[0][2] += 1
    V2[1][2] += 1; V2[3][2] -= 1
    W2[1][2] += 1; W2[3][2] += 1
    # P3 = d(g-e)
    U2[3][3] += 1
    V2[2][3] += 1; V2[0][3] -= 1
    W2[0][3] += 1; W2[2][3] += 1
    # P4 = (a+b)h
    U2[0][4] += 1; U2[1][4] += 1
    V2[3][4] += 1
    W2[0][4] -= 1; W2[1][4] += 1
    # P5 = (c-a)(e+f)
    U2[2][5] += 1; U2[0][5] -= 1
    V2[0][5] += 1; V2[1][5] += 1
    W2[3][5] += 1
    # P6 = (b-d)(g+h)
    U2[1][6] += 1; U2[3][6] -= 1
    V2[2][6] += 1; V2[3][6] += 1
    W2[0][6] += 1
    
    # Now build the 4x4x4 decomposition by composing Strassen 2x2x2.
    # The 4x4x4 decomposition has rank 49.
    # The structure: we partition A, B, C into 2x2 blocks of 2x2 submatrices.
    # A has 4x4 = 16 entries, indexed as (i,j) where i,j in {0,1,2,3}.
    # Block A[0,0] = A[0:2, 0:2], A[0,1] = A[0:2, 2:4], A[1,0] = A[2:4, 0:2], A[1,1] = A[2:4, 2:4]
    # Similarly for B and C.
    
    # The Strassen 4x4x4 decomposition:
    # C[0,0] = P0 * A_sub[0,0] * B_sub[0,0] ... no, that's not right.
    # Actually, the Strassen algorithm for 4x4x4 works as follows:
    # We compute 7 products of 2x2 matrices, where each 2x2 product is itself computed via Strassen 2x2x2.
    # The 7 products are:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # Each M_k is a 2x2 matrix. We compute each M_k using the Strassen 2x2x2 decomposition.
    # So for each k, we have 7 scalar multiplications (from the inner Strassen).
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # The U, V, W for the 4x4x4 decomposition:
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    # The 49 columns are indexed by (k, l) where k in {0..6} (outer Strassen) and l in {0..6} (inner Strassen).
    # For a given (k, l), the U entry for A index (i, j) is:
    #   U[i*4+j, k*7+l] = U2[i%2*2+j%2, l] * (coefficient from outer Strassen for block (i//2, j//2) in product k)
    # Similarly for V and W.
    
    # Let's define the outer Strassen coefficients.
    # For product k, the A block coefficient for block (bi, bj) and the B block coefficient for block (ci, cj).
    # Actually, it's simpler to just build it step by step.
    
    # Let's define the outer Strassen products more explicitly.
    # For each outer product k, we have linear combinations of A blocks and B blocks.
    # A blocks: A00, A01, A10, A11 (each is 2x2)
    # B blocks: B00, B01, B10, B11 (each is 2x2)
    
    # Outer product definitions (A_block_coeffs, B_block_coeffs):
    # k=0: A: {A00:1, A11:1}, B: {B00:1, B11:1}
    # k=1: A: {A10:1, A11:1}, B: {B00:1}
    # k=2: A: {A00:1}, B: {B01:1, B11:-1}
    # k=3: A: {A11:1}, B: {B10:1, B00:-1}
    # k=4: A: {A00:1, A01:1}, B: {B11:1}
    # k=5: A: {A10:1, A00:-1}, B: {B00:1, B01:1}
    # k=6: A: {A01:1, A11:-1}, B: {B10:1, B11:1}
    
    # For each outer product k, and for each inner Strassen product l (0..6),
    # we need to compute the contribution to U, V, W.
    
    # Let's build U4, V4, W4 for the 4x4x4 case.
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Define the outer Strassen structure.
    # For each k, we have a list of (a_block, coeff_a) and (b_block, coeff_b).
    # a_block is (bi, bj) where bi, bj in {0,1}
    # b_block is (ci, cj) where ci, cj in {0,1}
    
    outer_products = [
        # k=0
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # k=1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # k=2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # k=3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # k=4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # k=5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # k=6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    for k, (a_blocks, b_blocks) in enumerate(outer_products):
        for l in range(7):
            col = k * 7 + l
            # For each A block in a_blocks, add the inner Strassen U2 contribution.
            for (bi, bj, coeff_a) in a_blocks:
                # A block (bi, bj) corresponds to rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1.
                # The flat index for A[2*bi+i, 2*bj+j] is (2*bi+i)*4 + (2*bj+j).
                for i in range(2):
                    for j in range(2):
                        a_flat = (2*bi + i) * 4 + (2*bj + j)
                        U4[a_flat][col] += coeff_a * U2[i*2+j][l]
            
            # For each B block in b_blocks, add the inner Strassen V2 contribution.
            for (ci, cj, coeff_b) in b_blocks:
                # B block (ci, cj) corresponds to rows 2*ci, 2*ci+1 and cols 2*cj, 2*cj+1.
                # The flat index for B[2*ci+i, 2*cj+j] is (2*ci+i)*4 + (2*cj+j).
                for i in range(2):
                    for j in range(2):
                        b_flat = (2*ci + i) * 4 + (2*cj + j)
                        V4[b_flat][col] += coeff_b * V2[i*2+j][l]
            
            # For W, we need to compute the output block.
            # The output C block for product k is determined by the Strassen formula.
            # C00 = M0 - M4 + M6 + M2
            # C01 = M4 + M5
            # C10 = M1 + M6
            # C11 = M0 - M2 + M3 + M5
            
            # So for each output block (di, dj) in {0,1}x{0,1}, we need to sum contributions from various k.
            # Let's define the W coefficients for each output block and each k.
            # w_coeff[di][dj][k]
            
            # We'll handle W separately after building U and V.
    
    # Now build W4.
    # For each output block (di, dj), and for each inner Strassen product l,
    # we need to sum over k the contributions.
    
    # The Strassen formula for C blocks:
    # C00 = M0 - M4 + M6 + M2
    # C01 = M4 + M5
    # C10 = M1 + M6
    # C11 = M0 - M2 + M3 + M5
    
    # So for output block (di, dj), the coefficient for M_k is:
    w_coeff = [[0]*7 for _ in range(4)]  # w_coeff[di*2+dj][k]
    # C00: di=0, dj=0
    w_coeff[0][0] += 1  # M0
    w_coeff[0][4] += -1  # M4
    w_coeff[0][6] += 1  # M6
    w_coeff[0][2] += 1  # M2
    # C01: di=0, dj=1
    w_coeff[1][4] += 1  # M4
    w_coeff[1][5] += 1  # M5
    # C10: di=1, dj=0
    w_coeff[2][1] += 1  # M1
    w_coeff[2][6] += 1  # M6
    # C11: di=1, dj=1
    w_coeff[3][0] += 1  # M0
    w_coeff[3][2] += -1  # M2
    w_coeff[3][3] += 1  # M3
    w_coeff[3][5] += 1  # M5
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            for di in range(2):
                for dj in range(2):
                    out_block = di * 2 + dj
                    coeff_w = w_coeff[out_block][k]
                    if coeff_w != 0:
                        # Output block (di, dj) corresponds to rows 2*di, 2*di+1 and cols 2*dj, 2*dj+1.
                        # The flat index for C[2*di+i, 2*dj+j] is (2*di+i)*4 + (2*dj+j).
                        # But wait, the W index in the task is k*n + i, where k is the column index and i is the row index.
                        # For the 4x4x4 case, the output C is 4x4, so the flat index for C[i, j] is i*4 + j.
                        # But the task says W[k*n+i, r] where k is the column and i is the row.
                        # So for C[i, j], the W index is j*n + i.
                        # For the 4x4x4 case, n=4, so W index for C[i, j] is j*4 + i.
                        for i in range(2):
                            for j in range(2):
                                c_flat = (2*dj + j) * 4 + (2*di + i)  # j*4 + i
                                W4[c_flat][col] += coeff_w * W2[i*2+j][l]
    
    # Now we have U4, V4, W4 for the 4x4x4 case.
    # We need to extend this to 4x4x8.
    # The 4x4x8 multiplication can be seen as two 4x4x4 multiplications.
    # A is 4x4, B is 4x8, C is 4x8.
    # B = [B_left, B_right] where B_left is 4x4 and B_right is 4x4.
    # C = [C_left, C_right] where C_left = A * B_left and C_right = A * B_right.
    
    # The total rank is 98.
    # U has shape (16, 98), V has shape (32, 98), W has shape (32, 98).
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # For the left half (columns 0-3 of B, columns 0-3 of C):
    # U is the same as U4.
    # V is V4, but only for the left half of B (rows 0-3 of B, which is the first 4 rows of B).
    # W is W4, but for the left half of C (columns 0-3 of C, which is the first 4 columns of C).
    
    # For the right half (columns 4-7 of B, columns 4-7 of C):
    # U is the same as U4.
    # V is V4, but for the right half of B (rows 4-7 of B, which is the next 4 rows of B).
    # W is W4, but for the right half of C (columns 4-7 of C, which is the next 4 columns of C).
    
    # Let's fill in U, V, W.
    for r in range(16):
        for col in range(49):
            U[r][col] = U4[r][col]
            U[r][col + 49] = U4[r][col]
    
    for r in range(16):
        for col in range(49):
            # Left half of B
            V[r][col] = V4[r][col]
            # Right half of B
            V[r + 16][col + 49] = V4[r][col]
    
    for r in range(16):
        for col in range(49):
            # Left half of C
            # W index for C[i, j] where i in 0..3, j in 0..3 is j*4 + i.
            # In W4, the index is also j*4 + i.
            W[r][col] = W4[r][col]
            # Right half of C
            # W index for C[i, j] where i in 0..3, j in 4..7 is j*4 + i.
            # So for j' = j - 4, the index is (j'+4)*4 + i = j'*4 + i + 16.
            W[r + 16][col + 49] = W4[r][col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
