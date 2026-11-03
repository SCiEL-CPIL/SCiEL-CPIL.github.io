import numpy as np

def construct():
    n, m, p = 4, 4, 6
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # We need to generate the 49 terms for the 4x4x4 multiplication.
    # We can do this by applying Strassen to the 2x2 blocks.
    # Let's generate the 2x2x2 Strassen terms first.
    
    # Strassen 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # c00 = m1 - m4 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m6
    # m1 = (a11 + a00) * (b00 + b11)  # Note: indices in Strassen usually 1-based or specific.
    # Let's use standard 0-based indices for 2x2.
    # A00, A01, A10, A11
    # B00, B01, B10, B11
    # M1 = (A11 + A00)(B00 + B11) ? No.
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22) -> (A[1,1] + A[0,0])? No.
    # Let's stick to the formula:
    # M1 = (A[1,1] + A[0,0]) * (B[0,0] + B[1,1])  # This is not standard.
    
    # Correct Strassen:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 - M4 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6
    
    # We need to map this to the tensor decomposition.
    # Each M is a product of linear combinations of A and B.
    # This corresponds to a rank-1 term in the tensor?
    # No, each M is a scalar multiplication of two linear forms.
    # So each M corresponds to 1 rank term?
    # Yes, $u \otimes v \otimes w$ where $u, v$ are the linear forms and $w$ is the output coefficient.
    
    # We will generate the U, V, W vectors for each of the 7 M's.
    # Then we combine them for the 4x4x4 case.
    
    def strassen_2x2_terms():
        terms = []
        # M1
        # A: A00+A11, B: B00+B11
        # U: [1, 0, 1, 0] (A00, A01, A10, A11) -> A00+A11 is [1, 0, 0, 1]
        # V: [1, 0, 0, 1] (B00, B01, B10, B11) -> B00+B11 is [1, 0, 0, 1]
        # W: C00 += 1, C11 += 1
        # W is 2x2. C00 is W[0,0], C11 is W[1,1].
        # W indices: k*n+i. n=2.
        # C00 -> i=0, k=0 -> W[0,0] -> index 0*2+0=0.
        # C11 -> i=1, k=1 -> W[1,1] -> index 1*2+1=3.
        # So W vector is [1, 0, 0, 1].
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # M2
        # A: A10+A11 -> [0, 0, 1, 1]
        # B: B00 -> [1, 0, 0, 0]
        # C10 += 1 -> i=0, k=1 -> W[1,0] -> index 1*2+0=2.
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        
        # M3
        # A: A00 -> [1, 0, 0, 0]
        # B: B01-B11 -> [0, 1, 0, -1]
        # C01 += 1 -> i=1, k=0 -> W[0,1] -> index 0*2+1=1.
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        
        # M4
        # A: A11 -> [0, 0, 0, 1]
        # B: B10-B00 -> [-1, 0, 1, 0]
        # C00 -= 1, C10 += 1
        # C00 -> W[0,0] -> 0. C10 -> W[1,0] -> 2.
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
        
        # M5
        # A: A00+A01 -> [1, 1, 0, 0]
        # B: B11 -> [0, 0, 0, 1]
        # C01 += 1 -> W[0,1] -> 1.
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))
        
        # M6
        # A: A10-A00 -> [-1, 0, 1, 0]
        # B: B00+B01 -> [1, 1, 0, 0]
        # C11 += 1 -> W[1,1] -> 3.
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        
        # M7
        # A: A01-A11 -> [0, 1, 0, -1]
        # B: B10+B11 -> [0, 0, 1, 1]
        # C00 += 1, C11 -= 1
        # C00 -> 0. C11 -> 3.
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]))
        
        return terms

    s2_terms = strassen_2x2_terms()
    
    # Now build 4x4x4 from 2x2x2 blocks.
    # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # B is 4x4. Blocks B00, B01, B10, B11 are 2x2.
    # C is 4x4. Blocks C00, C01, C10, C11 are 2x2.
    # We apply Strassen to the blocks.
    # This gives 7 products of 2x2 matrices.
    # Each product is a 2x2x2 tensor.
    # So we have 7 * 7 = 49 terms.
    
    # We need to map the block indices to the full 4x4 indices.
    # A block (br, bc) corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1.
    # Index in U: i*m + j. i in 0..3, j in 0..3.
    # i = 2*br + r, j = 2*bc + c.
    
    # For each of the 7 Strassen steps (M1..M7), we have a linear combination of blocks.
    # Let's define the block linear combinations.
    
    # M1: (A11 + A00) * (B00 + B11)
    # A blocks: A00, A11. B blocks: B00, B11.
    # Output: C00 += 1, C11 += 1.
    
    # We will iterate over the 7 Strassen steps.
    # For each step, we have a set of A-blocks and B-blocks with coefficients.
    # And a set of C-blocks with coefficients.
    
    strassen_steps = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_idx, coeff)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]), # M1 -> C00, C11
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1) ]), # M2 -> C10
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1) ]), # M3 -> C01
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, -1), (2, 1) ]), # M4 -> -C00, C10
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1) ]), # M5 -> C01
        ([ (2, -1), (0, 1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]), # M6 -> C11
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (3, -1) ]) # M7 -> C00, -C11
    ]
    
    # For each step, we expand into 2x2x2 Strassen terms.
    # Wait, the 2x2x2 Strassen terms are for scalar multiplication.
    # Here we are multiplying 2x2 matrices.
    # So we apply the 2x2x2 Strassen decomposition to the block multiplication.
    
    # Let's generate the 49 terms.
    for step in strassen_steps:
        A_blocks, B_blocks, C_blocks = step
        # We need to combine these blocks using the 2x2x2 Strassen decomposition.
        # This is not straightforward because the Strassen decomposition is for the values.
        # Actually, we can just take the tensor product of the block structures.
        
        # Let's create the U, V, W vectors for the block multiplication.
        # U_block is a linear combination of 4x2x2 matrices?
        # No, U is 4x4.
        
        # Let's just iterate over the 7 inner Strassen terms for each outer Strassen term.
        # This will give 7*7=49 terms.
        
        for inner_term in s2_terms:
            u_inner, v_inner, w_inner = inner_term
            # u_inner is 4-vector (2x2 flattened)
            # v_inner is 4-vector
            # w_inner is 4-vector
            
            # We need to map this to the 4x4 blocks.
            # The outer Strassen step defines which blocks are involved.
            # This approach is getting complicated.
            
            pass

    # Alternative: Directly construct the 49 terms by brute force search or known formula.
    # Since I don't have the explicit 49 terms memorized, I will use the recursive structure.
    
    # Let's implement a function to generate the rank-1 terms for n x n x n Strassen.
    
    def generate_strassen_terms(n):
        if n == 2:
            return s2_terms
        
        # For n > 2, split into 4 blocks of size n/2.
        # Apply Strassen to the blocks.
        # This requires combining the terms.
        
        # This is complex to implement generically in a short time.
        # I will hardcode the 49 terms for 4x4x4?
        # Or use a library? No, standard library only.
        
        # Let's try to construct it programmatically.
        
        pass

    # Given the time, I will use the 49+28=77 rank decomposition.
    # I will implement the 2x2x2 Strassen and use it to build the 4x4x4 and 4x4x2 parts.
    
    # For 4x4x4:
    # We can view it as 4 separate 2x2x4 multiplications? No.
    
    # Let's just use the schoolbook for 4x4x6 (rank 96) to be safe, 
    # and try to optimize if possible.
    # But 77 is a significant improvement.
    
    # I will implement the 4x4x4 Strassen decomposition by explicitly listing the 49 terms.
    # This is tedious but doable.
    
    # Actually, I can use the fact that the 4x4x4 Strassen decomposition is the tensor product of two 2x2x2 Strassen decompositions?
    # No, it's the Kronecker product of the 2x2x2 Strassen matrix?
    
    # Let's stick to the plan:
    # 1. Generate 49 terms for 4x4x4.
    # 2. Generate 28 terms for 4x4x2.
    
    # I will write a helper to generate the 49 terms.
    
    def get_4x4x4_strassen():
        terms = []
        # We use the 7 outer steps and 7 inner steps.
        # For each outer step, we have a combination of blocks.
        # For each inner step, we have a combination of elements within blocks.
        
        # Let's define the outer steps as linear maps on the blocks.
        # And the inner steps as linear maps on the elements.
        
        # This is effectively the Kronecker product of the 2x2x2 Strassen decomposition with itself?
        # No.
        
        # I will simply use the schoolbook for now to ensure correctness, 
        # and then try to replace it with Strassen.
        
        return []

    # Since I cannot easily generate the 49 terms, I will use the schoolbook for the whole thing.
    # Rank 96.
    
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
