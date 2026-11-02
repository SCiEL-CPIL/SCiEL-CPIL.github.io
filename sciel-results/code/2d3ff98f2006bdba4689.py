import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will construct U, V, W as lists of lists or numpy arrays.
    # The prompt implies integer/half-integer entries.
    # We will use integers.
    
    # Rank 49 for 4x4x4 part
    # Rank 16 for 4x4x1 part
    # Total rank 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # A is 4x4, B is 4x4 (first 4 columns of B)
    # Blocks:
    # A11 A12
    # A21 A22
    # B11 B12
    # B21 B22
    
    # Strassen 2x2x2 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to the 49 scalar multiplications.
    # Each Mx is a 2x2x2 multiplication, which has rank 7.
    # So we have 7 * 7 = 49 terms.
    
    # Let's define the 7 Strassen terms for 2x2x2 as a function that returns U, V, W contributions
    # for a specific block offset.
    
    def add_strassen_2x2(offset_A, offset_B, offset_C, base_col):
        # offset_A, offset_B, offset_C are (i, j) coordinates in the 4x4 grid
        # But since we are building 4x4x4, the "blocks" are 2x2.
        # The indices in U, V, W are global.
        
        # The 7 terms for 2x2x2:
        # U indices: (i, j) in 0..1, 0..1
        # V indices: (j, k) in 0..1, 0..1
        # W indices: (i, k) in 0..1, 0..1
        
        # Term 1: (A11+A22)(B11+B22)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (1,0):-1, (0,1):-1, (1,1):1
        
        # Actually, let's just hardcode the 49 terms.
        # It's safer to generate them programmatically.
        
        pass

    # This is getting complex to hardcode. Let's use a helper to add a term.
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen 2x2x2 decomposition
    # Indices for 2x2 blocks:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # Contribution to C:
    # C00 = M1 - M2 - M4 + M7  (Wait, standard Strassen formulas)
    # C00 = M1 - M2 - M4 + M7?
    # Let's recall:
    # C00 = M1 - M2 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # So we need to add the 7 M terms to U, V, and then add the linear combinations to W.
    # But the decomposition is a sum of rank-1 tensors.
    # Each M term is itself a rank-7 tensor.
    # So we expand each M term into 7 rank-1 tensors.
    
    # Let's define the 7 rank-1 tensors for 2x2x2 Strassen.
    # s1: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,0]=-1, W[0,1]=-1, W[1,1]=1
    # s2: U[1,0]=1, U[1,1]=1; V[0,0]=1; W[0,0]=-1, W[1,0]=1
    # s3: U[0,0]=1; V[0,1]=1, V[1,1]=-1; W[0,1]=1
    # s4: U[1,1]=1; V[1,0]=1, V[0,0]=-1; W[1,0]=1
    # s5: U[0,0]=1, U[0,1]=1; V[1,1]=1; W[0,1]=1
    # s6: U[1,0]=1, U[0,0]=-1; V[0,0]=1, V[0,1]=1; W[1,1]=-1
    # s7: U[0,1]=1, U[1,1]=-1; V[1,0]=1, V[1,1]=1; W[1,1]=1
    
    # We need to map these local indices to global indices.
    # For the 4x4x4 part, we have 4 blocks of 2x2.
    # Block (0,0): A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    # Block (0,1): A rows 0-1, cols 2-3; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    # Block (1,0): A rows 2-3, cols 0-1; B rows 2-3, cols 0-1; C rows 2-3, cols 0-1
    # Block (1,1): A rows 2-3, cols 2-3; B rows 2-3, cols 2-3; C rows 2-3, cols 2-3
    
    # Wait, Strassen computes C = A B.
    # The 4x4x4 multiplication involves 4 such block multiplications?
    # No, Strassen for 4x4x4 uses the 2x2 block structure of the 4x4 matrices.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = [[C11, C12], [C21, C22]] where each is 2x2.
    # The 7 M terms are 2x2x2 multiplications.
    # M1 = (A11+A22)(B11+B22)
    # This M1 is a 2x2 matrix.
    # To compute M1, we use the rank-7 decomposition of 2x2x2.
    # So for each of the 7 M terms, we add 7 rank-1 tensors.
    # Total 49.
    
    # We need to define the linear combinations for A and B blocks in the M terms.
    # And the linear combinations for C blocks in the result.
    
    # Let's define the 7 M terms' contributions to U, V, W.
    # U indices are (i, j) in 4x4.
    # V indices are (j, k) in 4x4.
    # W indices are (i, k) in 4x4.
    
    # For M1 = (A11+A22)(B11+B22):
    # U part: A11 and A22.
    # A11 is at (0,0) to (1,1). A22 is at (2,2) to (3,3).
    # So U has 1s at (0,0), (0,1), (1,0), (1,1) and (2,2), (2,3), (3,2), (3,3).
    # V part: B11 and B22.
    # B11 is at (0,0) to (1,1). B22 is at (2,2) to (3,3).
    # So V has 1s at (0,0), (0,1), (1,0), (1,1) and (2,2), (2,3), (3,2), (3,3).
    # W part: C11 and C22.
    # C11 = M1 - M2 - M4 + M7.
    # C22 = M1 - M3 + M6 - M7.
    # So for M1, W has 1s at C11 and C22 blocks.
    # C11 is at (0,0) to (1,1). C22 is at (2,2) to (3,3).
    # But we need to distribute the 1s according to the 2x2x2 rank-7 decomposition.
    
    # This is getting complicated to do by hand.
    # Let's write a helper to add a 2x2x2 Strassen decomposition at a specific block location.
    
    def add_2x2_strassen(u_block, v_block, w_block, base_col):
        # u_block, v_block, w_block are 2x2 numpy arrays or lists
        # We add 7 terms.
        nonlocal col
        
        # Term 1
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,0)=-1, (0,1)=-1, (1,1)=1
        u_idx = [u_block[0][0], u_block[1][1]]
        v_idx = [v_block[0][0], v_block[1][1]]
        w_idx = [w_block[0][0], w_block[1][0], w_block[0][1], w_block[1][1]]
        u_val = [1, 1]
        v_val = [1, 1]
        w_val = [1, -1, -1, 1]
        # This is not a single rank-1 term. It's a sum of 4 rank-1 terms?
        # No, the rank-1 term is defined by U, V, W vectors.
        # U vector has non-zeros at (0,0) and (1,1).
        # V vector has non-zeros at (0,0) and (1,1).
        # W vector has non-zeros at (0,0), (1,0), (0,1), (1,1).
        # So we add ONE term to U, V, W.
        
        # Let's implement a generic add_term for a single rank-1 tensor.
        
        pass

    # This manual construction is error-prone.
    # Let's use a known library or a simpler approach.
    # Since I cannot use external libraries other than numpy/scipy, I will hardcode the 49 terms.
    # Or, I can use the fact that the rank of 4x4x4 is 49 and just use the schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x1 (rank 16), total 80.
    # But 65 is better.
    
    # Let's try to construct the 49 terms.
    
    # The 7 Strassen terms for 2x2x2:
    # T1: U=[1,0,0,1], V=[1,0,0,1], W=[1,-1,-1,1] (indices 00, 10, 01, 11)
    # T2: U=[0,0,1,1], V=[1,0,0,0], W=[-1,1,0,0]
    # T3: U=[1,0,0,0], V=[0,1,-1,0], W=[0,1,0,0]
    # T4: U=[0,0,0,1], V=[-1,0,1,0], W=[0,0,1,0]
    # T5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    # T6: U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,0,-1]
    # T7: U=[0,1,0,-1], V=[0,0,1,1], W=[0,0,0,1]
    
    # Note: Indices are flattened 0..3 for 2x2.
    # 00=0, 01=1, 10=2, 11=3.
    
    # For 4x4x4, we have 4 blocks.
    # Block (0,0): U indices 0..3, V indices 0..3, W indices 0..3
    # Block (0,1): U indices 4..7, V indices 8..11, W indices 4..7
    # Block (1,0): U indices 8..11, V indices 4..7, W indices 8..11
    # Block (1,1): U indices 12..15, V indices 12..15, W indices 12..15
    
    # Wait, the blocks are:
    # A11: rows 0-1, cols 0-1 -> U indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> U indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> U indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> U indices 10,11,14,15
    
    # B11: rows 0-1, cols 0-1 -> V indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> V indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> V indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> V indices 10,11,14,15
    
    # C11: rows 0-1, cols 0-1 -> W indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> W indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> W indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> W indices 10,11,14,15
    
    # The 7 M terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to add the 7 rank-1 tensors for each M term.
    
    # Let's define the 7 rank-1 tensors for 2x2x2.
    # T1: U=[1,0,0,1], V=[1,0,0,1], W=[1,-1,-1,1]
    # T2: U=[0,0,1,1], V=[1,0,0,0], W=[-1,1,0,0]
    # T3: U=[1,0,0,0], V=[0,1,-1,0], W=[0,1,0,0]
    # T4: U=[0,0,0,1], V=[-1,0,1,0], W=[0,0,1,0]
    # T5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    # T6: U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,0,-1]
    # T7: U=[0,1,0,-1], V=[0,0,1,1], W=[0,0,0,1]
    
    # For M1, U is A11+A22.
    # A11 is at U indices 0,1,4,5. A22 is at U indices 10,11,14,15.
    # So U vector for M1 has 1s at these indices.
    # But we need to apply the T1..T7 patterns.
    # T1 pattern: U has 1s at (0,0) and (1,1) of the 2x2 block.
    # So for A11, U indices 0 and 5. For A22, U indices 10 and 15.
    # So U vector for M1-T1 has 1s at 0, 5, 10, 15.
    # Similarly for V and W.
    
    # This is doable.
    
    # Let's write a function to add a term.
    
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Define the 7 patterns for 2x2x2.
    # Patterns are lists of (index, value) for U, V, W.
    # Indices are 0..3 for 2x2.
    
    patterns = [
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (1,-1), (2,-1), (3,1) ]),
        ([ (2,1), (3,1) ], [ (0,1) ], [ (0,-1), (1,1) ]),
        ([ (0,1) ], [ (1,1), (2,-1) ], [ (1,1) ]),
        ([ (3,1) ], [ (0,-1), (2,1) ], [ (2,1) ]),
        ([ (0,1), (1,1) ], [ (3,1) ], [ (1,1) ]),
        ([ (1,-1), (2,1) ], [ (0,1), (1,1) ], [ (3,-1) ]),
        ([ (1,1), (3,-1) ], [ (2,1), (3,1) ], [ (3,1) ])
    ]
    
    # Map block names to global indices.
    # A11: [0, 1, 4, 5]
    # A12: [2, 3, 6, 7]
    # A21: [8, 9, 12, 13]
    # A22: [10, 11, 14, 15]
    
    # B11: [0, 1, 4, 5]
    # B12: [2, 3, 6, 7]
    # B21: [8, 9, 12, 13]
    # B22: [10, 11, 14, 15]
    
    # C11: [0, 1, 4, 5]
    # C12: [2, 3, 6, 7]
    # C21: [8, 9, 12, 13]
    # C22: [10, 11, 14, 15]
    
    # M1 = (A11+A22)(B11+B22)
    # U blocks: A11, A22
    # V blocks: B11, B22
    # W blocks: C11, C22
    
    # We need to combine the patterns with the block indices.
    
    def get_block_indices(block_name):
        if block_name == 'A11': return [0, 1, 4, 5]
        if block_name == 'A12': return [2, 3, 6, 7]
        if block_name == 'A21': return [8, 9, 12, 13]
        if block_name == 'A22': return [10, 11, 14, 15]
        if block_name == 'B11': return [0, 1, 4, 5]
        if block_name == 'B12': return [2, 3, 6, 7]
        if block_name == 'B21': return [8, 9, 12, 13]
        if block_name == 'B22': return [10, 11, 14, 15]
        if block_name == 'C11': return [0, 1, 4, 5]
        if block_name == 'C12': return [2, 3, 6, 7]
        if block_name == 'C21': return [8, 9, 12, 13]
        if block_name == 'C22': return [10, 11, 14, 15]
    
    def add_strassen_term(u_blocks, v_blocks, w_blocks, signs_u, signs_v, signs_w):
        # u_blocks: list of block names
        # signs_u: list of signs (+1 or -1)
        # Same for v, w
        
        for p_idx, (u_pat, v_pat, w_pat) in enumerate(patterns):
            u_indices = []
            u_vals = []
            for b_idx, block in enumerate(u_blocks):
                ind = get_block_indices(block)
                sign = signs_u[b_idx]
                for local_idx, val in u_pat:
                    u_indices.append(ind[local_idx])
                    u_vals.append(val * sign)
            
            v_indices = []
            v_vals = []
            for b_idx, block in enumerate(v_blocks):
                ind = get_block_indices(block)
                sign = signs_v[b_idx]
                for local_idx, val in v_pat:
                    v_indices.append(ind[local_idx])
                    v_vals.append(val * sign)
            
            w_indices = []
            w_vals = []
            for b_idx, block in enumerate(w_blocks):
                ind = get_block_indices(block)
                sign = signs_w[b_idx]
                for local_idx, val in w_pat:
                    w_indices.append(ind[local_idx])
                    w_vals.append(val * sign)
            
            add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals)
    
    # M1 = (A11+A22)(B11+B22)
    add_strassen_term(['A11', 'A22'], ['B11', 'B22'], ['C11', 'C22'], [1, 1], [1, 1], [1, 1])
    
    # M2 = (A21+A22)B11
    add_strassen_term(['A21', 'A22'], ['B11'], ['C11', 'C21'], [1, 1], [1], [-1, 1])
    
    # M3 = A11(B12-B22)
    add_strassen_term(['A11'], ['B12', 'B22'], ['C12'], [1], [1, -1], [1])
    
    # M4 = A22(B21-B11)
    add_strassen_term(['A22'], ['B21', 'B11'], ['C11', 'C21'], [1], [1, -1], [-1, 1])
    
    # M5 = (A11+A12)B22
    add_strassen_term(['A11', 'A12'], ['B22'], ['C12'], [1, 1], [1], [1])
    
    # M6 = (A21-A11)(B11+B12)
    add_strassen_term(['A21', 'A11'], ['B11', 'B12'], ['C22'], [1, -1], [1, 1], [1])
    
    # M7 = (A12-A22)(B21+B22)
    add_strassen_term(['A12', 'A22'], ['B21', 'B22'], ['C11', 'C22'], [1, -1], [1, 1], [1, -1])
    
    # Part 2: 4x4x1 multiplication (Rank 16)
    # A is 4x4, B is 4x1 (last column of B)
    # C is 4x1 (last column of C)
    # C_i = sum_j A_ij B_j
    # This is 16 scalar multiplications.
    
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4  # Last column of B
            w_idx = 4 * 4 + i  # Last column of C (k=4, i=0..3)
            add_term([u_idx], [v_idx], [w_idx], [1], [1], [1])
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
