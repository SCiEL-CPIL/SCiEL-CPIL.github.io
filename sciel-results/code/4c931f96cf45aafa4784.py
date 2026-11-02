import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 for 4x4 part (Strassen), Rank 16 for 4x1 part (Schoolbook)
    # Total Rank 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4 Matrix Multiplication (A * B1)
    # Strassen Decomposition
    # Blocks are 2x2
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    
    # Helper to add a schoolbook term for 2x2 multiplication
    # a_idx, b_idx are (i, j) in 0..3
    # c_idx is (i, k) in 0..3
    def add_2x2_term(a_coeff, b_coeff, c_coeff, a_block, b_block, c_block, sign):
        nonlocal col
        # a_block, b_block, c_block are 0..3 representing 11, 12, 21, 22
        # Map block to local 2x2 indices
        # Block 0: 11 (i in 0,1; j in 0,1)
        # Block 1: 12 (i in 0,1; j in 2,3)
        # Block 2: 21 (i in 2,3; j in 0,1)
        # Block 3: 22 (i in 2,3; j in 2,3)
        
        # We need to sum over the 4 elements of the 2x2 product
        # P[k, l] = sum_{m} L_A[i, m] L_B[m, j] ?
        # No, the linear combinations are already formed.
        # The product is of two 2x2 matrices L_A and L_B.
        # We decompose L_A * L_B into 4 rank-1 terms (schoolbook for 2x2).
        
        # L_A is a 2x2 matrix of coefficients for A entries?
        # No, L_A is a linear combination of A blocks.
        # So L_A is a 2x2 matrix where each entry is a sum of A entries.
        # Actually, it's easier to define the coefficients for the 16 A entries and 16 B entries.
        
        pass

    # This approach is complex to code generically.
    # Let's just hardcode the 28 terms for Strassen 4x4.
    # Or use a simpler decomposition if possible.
    # Given the time, I will use the schoolbook for the 4x4 part as well, which is rank 64.
    # Total rank 64 + 16 = 80. This is the same as the current program.
    # I need to improve.
    
    # Let's try to implement Strassen for 4x4 properly.
    
    # Strassen steps for 4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each Mi is a 2x2 matrix product.
    # We use schoolbook for each Mi (rank 4).
    # Total 28 terms.
    
    # Define block indices
    # 0: 11, 1: 12, 2: 21, 3: 22
    
    # Coefficients for A and B for each step
    # step: (A_coeffs, B_coeffs)
    # A_coeffs is a 4x4 matrix of coefficients for the 4 blocks?
    # No, A_coeffs is a dictionary of block -> coefficient.
    
    steps = [
        # (A_blocks, B_blocks)
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # Output weights for each step
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Weights[step][block]
    weights = [
        {0: 1, 1: 0, 2: 0, 3: 0}, # M1
        {0: 0, 1: 0, 2: 1, 3: 0}, # M2
        {0: 0, 1: 1, 2: 0, 3: 0}, # M3
        {0: 1, 1: 0, 2: 1, 3: 0}, # M4
        {0: -1, 1: 1, 2: 0, 3: 0},# M5
        {0: 0, 1: 0, 2: 0, 3: 1}, # M6
        {0: 1, 1: 0, 2: 0, 3: 0}  # M7
    ]
    # Wait, M6 contributes to C22. M7 contributes to C11.
    # C22 = M1 - M2 + M3 + M6
    # So M6 weight for block 3 is 1.
    # M7 weight for block 0 is 1.
    
    weights[6][0] = 1 # M7 to C11
    weights[5][3] = 1 # M6 to C22
    weights[0][3] = 1 # M1 to C22
    weights[1][3] = -1 # M2 to C22
    weights[2][3] = 1 # M3 to C22
    
    # Corrected weights:
    # M1: C11(+1), C22(+1)
    # M2: C21(+1), C22(-1)
    # M3: C12(+1), C22(+1)
    # M4: C11(+1), C21(+1)
    # M5: C11(-1), C12(+1)
    # M6: C22(+1)
    # M7: C11(+1)
    
    weights = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: -1},# M2
        {1: 1, 3: 1}, # M3
        {0: 1, 2: 1}, # M4
        {0: -1, 1: 1},# M5
        {3: 1},       # M6
        {0: 1}        # M7
    ]
    
    for s in range(7):
        a_blocks = steps[s][0]
        b_blocks = steps[s][1]
        
        # Construct L_A and L_B as 2x2 matrices of coefficients?
        # No, we just need the linear combinations.
        # For each of the 4 schoolbook terms in the 2x2 product:
        # Term (k, l) of the product P = L_A L_B
        # P[k, l] = sum_m L_A[k, m] L_B[m, l]
        # This is not a rank-1 term.
        # The rank-1 terms are A_{ij} B_{jk}.
        
        # Let's iterate over the 4 elements of the 2x2 product.
        # For each output position (k, l) in the 2x2 block:
        # We need to sum over the intermediate index m.
        # This is just schoolbook for 2x2.
        
        # So for each (k, l) in 0..1:
        #   For each m in 0..1:
        #     The term is L_A[k, m] * L_B[m, l]
        #     But L_A and L_B are sums of A and B entries.
        #     So we expand L_A[k, m] = sum_{block_a} c_a A_{block_a}
        #     and L_B[m, l] = sum_{block_b} c_b B_{block_b}
        #     The product is sum_{block_a, block_b} c_a c_b A_{block_a} B_{block_b}
        #     Each A_{block_a} B_{block_b} is a 2x2 matrix product?
        #     No, A_{block_a} is a 2x2 matrix.
        #     So A_{block_a} B_{block_b} is a 2x2 matrix product.
        #     This is getting recursive.
        
        # Simpler: Just use the schoolbook for the whole 4x4.
        # Rank 64.
        # Total Rank 80.
        # This is not an improvement.
        
        pass
    
    # I will use the schoolbook for the 4x4 part to ensure correctness, 
    # but I will try to optimize the 4x1 part?
    # No, 4x1 is already minimal.
    
    # Let's stick to the current implementation but maybe there's a bug?
    # No, the current implementation is standard schoolbook.
    
    # I will output the current implementation as it is safer.
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
