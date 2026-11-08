"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition that splits the inner dimension m=3 into 2+1.
    # A is 2x3, split into A1 (2x2) and A2 (2x1).
    # B is 3x7, split into B1 (2x7) and B2 (1x7).
    # C = A1 B1 + A2 B2.
    
    # A1 B1: 2x2 * 2x7.
    # Split B1 (2x7) into B1_1 (2x2), B1_2 (2x2), B1_3 (2x2), B1_4 (2x1).
    # For each 2x2 block, use Strassen (rank 7).
    # For the 2x1 block, use Schoolbook (rank 4).
    
    # A2 B2: 2x1 * 1x7.
    # Schoolbook rank 14.
    
    # Total rank = 3 * 7 + 4 + 14 = 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)] # 6 x 39
    V = [[0] * rank for _ in range(m * p)] # 21 x 39
    W = [[0] * rank for _ in range(p * n)] # 14 x 39
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Indices:
    # A is 2x3. Rows 0,1. Cols 0,1,2.
    # A1 is rows 0,1 cols 0,1.
    # A2 is rows 0,1 col 2.
    
    # B is 3x7. Rows 0,1,2. Cols 0..6.
    # B1 is rows 0,1 cols 0..6.
    # B2 is row 2 cols 0..6.
    
    # Part 1: A2 B2 (2x1 * 1x7)
    # C[i, k] += A[2, i] * B[2, k]
    # A indices: i*m + j. A2 is j=2. So A[0,2] is 0*3+2=2, A[1,2] is 1*3+2=5.
    # B indices: j*p + k. B2 is j=2. So B[2, k] is 2*7+k = 14+k.
    # W indices: k*n + i. W[k, i] is k*2+i.
    
    for i in range(2):
        for k in range(7):
            u_idx = i * m + 2 # A[i, 2]
            v_idx = 2 * p + k # B[2, k]
            w_idx = k * n + i # W[k, i]
            add_term(u_idx, v_idx, w_idx, 1, 1, 1)
            
    # Part 2: A1 B1 (2x2 * 2x7)
    # B1 is split into 3 blocks of 2x2 and 1 block of 2x1.
    # Block 1: cols 0,1.
    # Block 2: cols 2,3.
    # Block 3: cols 4,5.
    # Block 4: col 6.
    
    # Strassen for 2x2 * 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M3 + M5
    
    def add_strassen_block(b_col_start):
        nonlocal col
        # b_col_start is the starting column index in B1 (which is 0..6)
        # B1 rows are 0,1.
        # B indices:
        # B00 is B[0, b_col_start] -> 0*7 + b_col_start
        # B01 is B[0, b_col_start+1] -> 0*7 + b_col_start+1
        # B10 is B[1, b_col_start] -> 1*7 + b_col_start
        # B11 is B[1, b_col_start+1] -> 1*7 + b_col_start+1
        
        # A indices (A1 is 2x2, rows 0,1 cols 0,1)
        # A00 is A[0,0] -> 0
        # A01 is A[0,1] -> 1
        # A10 is A[1,0] -> 3
        # A11 is A[1,1] -> 4
        
        # W indices for this block:
        # C00 is W[b_col_start, 0] -> b_col_start*2 + 0
        # C01 is W[b_col_start+1, 0] -> (b_col_start+1)*2 + 0
        # C10 is W[b_col_start, 1] -> b_col_start*2 + 1
        # C11 is W[b_col_start+1, 1] -> (b_col_start+1)*2 + 1
        
        b00 = 0 * p + b_col_start
        b01 = 0 * p + b_col_start + 1
        b10 = 1 * p + b_col_start
        b11 = 1 * p + b_col_start + 1
        
        w00 = b_col_start * n + 0
        w01 = (b_col_start + 1) * n + 0
        w10 = b_col_start * n + 1
        w11 = (b_col_start + 1) * n + 1
        
        # M1
        add_term(0, b00, w00, 1, 1, 1)
        add_term(4, b00, w00, 1, 1, 1)
        add_term(0, b11, w00, 1, 1, 1)
        add_term(4, b11, w00, 1, 1, 1)
        
        # M2
        add_term(3, b00, w10, 1, 1, 1)
        add_term(4, b00, w10, 1, 1, 1)
        
        # M3
        add_term(0, b10, w01, 1, 1, 1)
        add_term(0, b11, w01, 1, -1, 1)
        
        # M4
        add_term(4, b01, w11, 1, 1, 1)
        add_term(4, b00, w11, 1, -1, 1)
        
        # M5
        add_term(0, b11, w01, 1, 1, 1)
        add_term(1, b11, w01, 1, 1, 1)
        add_term(0, b11, w11, 1, 1, 1)
        add_term(1, b11, w11, 1, 1, 1)
        
        # M6
        add_term(3, b00, w00, 1, 1, 1)
        add_term(0, b00, w00, 1, -1, 1)
        add_term(3, b01, w00, 1, 1, 1)
        add_term(0, b01, w00, 1, -1, 1)
        
        # M7
        add_term(1, b10, w10, 1, 1, 1)
        add_term(4, b10, w10, 1, -1, 1)
        add_term(1, b11, w10, 1, 1, 1)
        add_term(4, b11, w10, 1, -1, 1)
        
        # Wait, the Strassen formulas combine these M's.
        # The above add_term calls are for the M's, but we need to add them to W with the correct coefficients.
        # My add_term function adds 1 term to the decomposition.
        # Strassen decomposes the product into 7 products.
        # Each product is a rank-1 tensor in the sub-space?
        # No, the decomposition is a sum of rank-1 tensors.
        # The 7 products M1..M7 are not rank-1 tensors in the original space.
        # They are intermediate values.
        # The rank of the tensor is the number of scalar multiplications.
        # So we need 7 terms for the 2x2x2 block.
        
        # Let's re-implement the Strassen terms correctly.
        # We need to define the 7 rank-1 tensors that sum to the 2x2x2 multiplication tensor.
        
        pass

    # The previous add_term calls in add_strassen_block were incorrect because they didn't form a valid rank-1 decomposition of the 2x2x2 tensor.
    # I will use the known rank-7 decomposition for 2x2x2.
    
    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A2 B2 (2x1 * 1x7) -> 14 terms
    for i in range(2):
        for k in range(7):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx, 1, 1, 1)
            
    # Part 2: A1 B1 (2x2 * 2x7)
    # 3 blocks of 2x2, 1 block of 2x1.
    
    # Strassen Rank-7 Decomposition for 2x2x2
    # T = sum M_l
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2: (A10+A11)B00 -> C10 += M2
    # M3: A00(B10-B11) -> C01 += M3
    # M4: A11(B01-B00) -> C11 -= M4
    # M5: (A00+A01)B11 -> C01 += M5, C11 += M5
    # M6: (A10-A00)(B00+B01) -> C00 += M6
    # M7: (A01-A11)(B10+B11) -> C10 += M7
    
    # We need to express this as a sum of rank-1 tensors.
    # Each M_l is a product of a linear form in A and a linear form in B.
    # So each M_l corresponds to one term in the decomposition.
    
    def add_strassen_block_v2(b_col_start):
        nonlocal col
        b00 = 0 * p + b_col_start
        b01 = 0 * p + b_col_start + 1
        b10 = 1 * p + b_col_start
        b11 = 1 * p + b_col_start + 1
        
        w00 = b_col_start * n + 0
        w01 = (b_col_start + 1) * n + 0
        w10 = b_col_start * n + 1
        w11 = (b_col_start + 1) * n + 1
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: W00, W11.
        add_term(0, b00, w00, 1, 1, 1)
        add_term(4, b00, w00, 1, 1, 1)
        add_term(0, b11, w00, 1, 1, 1)
        add_term(4, b11, w00, 1, 1, 1)
        # This is 4 terms, but M1 is one multiplication.
        # The rank is the number of multiplications.
        # So M1 contributes 1 to the rank?
        # No, the rank is the number of rank-1 tensors in the sum.
        # The decomposition is T = sum_{l=1}^7 L_A_l \otimes L_B_l \otimes L_C_l.
        # Each l corresponds to one term.
        # So we need 7 terms for the 2x2x2 block.
        
        pass

    # I will use the standard 7-term decomposition.
    # Term 1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1) ? No.
    # The vectors U, V, W in the decomposition are of length 4 (for 2x2).
    # Here U is length 6, V is length 21, W is length 14.
    # We just need to add 7 terms for each 2x2 block.
    
    # Let's hardcode the 7 terms for a 2x2x2 block.
    # A indices: 0,1,3,4 (A00, A01, A10, A11)
    # B indices: b00, b01, b10, b11
    # W indices: w00, w01, w10, w11
    
    def add_strassen_block_v3(b_col_start):
        nonlocal col
        b00 = 0 * p + b_col_start
        b01 = 0 * p + b_col_start + 1
        b10 = 1 * p + b_col_start
        b11 = 1 * p + b_col_start + 1
        
        w00 = b_col_start * n + 0
        w01 = (b_col_start + 1) * n + 0
        w10 = b_col_start * n + 1
        w11 = (b_col_start + 1) * n + 1
        
        # M1
        U[0][col] = 1; V[b00][col] = 1; W[w00][col] = 1
        U[4][col] = 1; V[b00][col] = 1; W[w00][col] = 1
        U[0][col] = 1; V[b11][col] = 1; W[w00][col] = 1
        U[4][col] = 1; V[b11][col] = 1; W[w00][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1; V[b00][col] = 1; W[w10][col] = 1
        U[4][col] = 1; V[b00][col] = 1; W[w10][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1; V[b10][col] = 1; W[w01][col] = 1
        U[0][col] = 1; V[b11][col] = -1; W[w01][col] = 1
        col += 1
        
        # M4
        U[4][col] = 1; V[b01][col] = 1; W[w11][col] = -1
        U[4][col] = 1; V[b00][col] = -1; W[w11][col] = -1
        col += 1
        
        # M5
        U[0][col] = 1; V[b11][col] = 1; W[w01][col] = 1
        U[1][col] = 1; V[b11][col] = 1; W[w01][col] = 1
        U[0][col] = 1; V[b11][col] = 1; W[w11][col] = 1
        U[1][col] = 1; V[b11][col] = 1; W[w11][col] = 1
        col += 1
        
        # M6
        U[3][col] = 1; V[b00][col] = 1; W[w00][col] = 1
        U[0][col] = -1; V[b00][col] = 1; W[w00][col] = 1
        U[3][col] = 1; V[b01][col] = 1; W[w00][col] = 1
        U[0][col] = -1; V[b01][col] = 1; W[w00][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1; V[b10][col] = 1; W[w10][col] = 1
        U[4][col] = -1; V[b10][col] = 1; W[w10][col] = 1
        U[1][col] = 1; V[b11][col] = 1; W[w10][col] = 1
        U[4][col] = -1; V[b11][col] = 1; W[w10][col] = 1
        col += 1

    # Call for 3 blocks
    add_strassen_block_v3(0)
    add_strassen_block_v3(2)
    add_strassen_block_v3(4)
    
    # Part 3: A1 B1_last (2x2 * 2x1) -> Rank 4
    b_col_start = 6
    b00 = 0 * p + b_col_start
    b10 = 1 * p + b_col_start
    
    w00 = b_col_start * n + 0
    w10 = b_col_start * n + 1
    
    # C00 = A00 B00 + A10 B10
    # C10 = A01 B00 + A11 B10
    # Wait, A is 2x2.
    # C00 = A00 B00 + A10 B10? No.
    # C is 2x1.
    # C0 = A00 B00 + A01 B10? No.
    # A is 2x2. B is 2x1.
    # C0 = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C1 = A[1,0]B[0,0] + A[1,1]B[1,0]
    
    add_term(0, b00, w00, 1, 1, 1)
    add_term(1, b10, w00, 1, 1, 1)
    add_term(3, b00, w10, 1, 1, 1)
    add_term(4, b10, w10, 1, 1, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
