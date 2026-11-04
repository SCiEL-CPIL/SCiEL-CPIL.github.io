import time
import numpy as np

def construct():
    t_start = time.time()
    deadline = t_start + 140
    
    n, m, p = 4, 4, 5
    
    # Rank 65 decomposition
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen 4x4x4 (rank 49)
    # We need to generate the 49 terms for 4x4 matrix multiplication.
    # This is done by recursively applying Strassen 2x2.
    
    # Helper to get Strassen 2x2 terms (u, v, w)
    # u is 4x1 (flattened 2x2 A)
    # v is 4x1 (flattened 2x2 B)
    # w is 4x1 (flattened 2x2 C)
    
    s_u = np.array([
        [1, 0, 0, 1], # M1: A11+A22
        [0, 1, 1, 0], # M2: A21+A22 (Note: index 1 is A12, 2 is A21, 3 is A22? 
                      # Standard flatten: 0:00, 1:01, 2:10, 3:11
                      # A11=0, A12=1, A21=2, A22=3
                      # M1: A11+A22 -> [1,0,0,1]
                      # M2: A21+A22 -> [0,0,1,1]
                      # M3: A11 -> [1,0,0,0]
                      # M4: A22 -> [0,0,0,1]
                      # M5: A11+A12 -> [1,1,0,0]
                      # M6: A21-A11 -> [-1,0,1,0]
                      # M7: A12-A22 -> [0,1,0,-1]
        ], dtype=float).T # Transpose to have 7 rows? No, 7 terms, 4 entries each.
        
    # Let's define the 7 terms explicitly for 2x2
    terms_2x2 = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1: (A11+A22)(B11+B22) -> C11, C22
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2: (A21+A22)B11 -> C21
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3: A11(B21-B22) -> C12
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 1]), # M4: A22(B12-B11) -> C11, C22
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5: (A11+A12)B22 -> -C11, C12
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6: (A21-A11)(B11+B12) -> C22
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7: (A12-A22)(B21+B22) -> C11
    ]
    
    # For 4x4, we have 4 blocks in A and 4 blocks in B.
    # Block indices: 00, 01, 10, 11
    # Strassen steps for 4x4:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00)
    # P3 = (A00)(B10-B11)
    # P4 = (A11)(B01-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pi is a 2x2 matrix product.
    # We need to map the 2x2 terms to the 4x4 indices.
    
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B blocks:
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C blocks:
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # We define the 7 Strassen steps for 4x4.
    # Each step k has a linear combination of A blocks and B blocks.
    # u_k is a 16x1 vector (flattened 4x4 A).
    # v_k is a 16x1 vector (flattened 4x4 B).
    # w_k is a 16x1 vector (flattened 4x4 C).
    
    # But wait, the Strassen algorithm computes 7 matrix products.
    # Each matrix product is 2x2.
    # So for each of the 7 steps, we have 7 scalar multiplications.
    # Total 49.
    
    # Let's define the 7 steps for 4x4.
    # Step 1: P1 = (A00+A11)(B00+B11)
    # u1 is A00+A11. v1 is B00+B11.
    # The result P1 contributes to C00 and C11.
    # Specifically, C00 += P1, C11 += P1.
    # So for each of the 7 terms in the 2x2 decomposition of P1,
    # the w vector will be in the C00 and C11 blocks.
    
    # This is getting complex to hardcode.
    # Let's use a simpler approach: 
    # Use the schoolbook for the first 4 columns? No, that's rank 64.
    # Schoolbook for 4x4x4 is rank 64.
    # Schoolbook for 4x4x1 is rank 16.
    # Total 80.
    
    # We need to reduce the 4x4x4 part.
    # Strassen 4x4x4 is rank 49.
    # So 49 + 16 = 65.
    
    # Let's implement the Strassen 4x4 decomposition by recursion.
    
    def get_strassen_2x2():
        return terms_2x2

    def expand_2x2_to_4x4(u_block, v_block, w_block, a_block_idx, b_block_idx, c_block_idx):
        # u_block is 4x1 (vector for one 2x2 block of A)
        # v_block is 4x1 (vector for one 2x2 block of B)
        # w_block is 4x1 (vector for one 2x2 block of C)
        # a_block_idx: which 2x2 block of A (0:00, 1:01, 2:10, 3:11)
        # b_block_idx: which 2x2 block of B
        # c_block_idx: which 2x2 block of C
        
        u_full = np.zeros(16)
        v_full = np.zeros(16)
        w_full = np.zeros(16)
        
        # Map block index to start index in flattened 4x4
        # 0: 0-3, 1: 4-7, 2: 8-11, 3: 12-15
        start_a = a_block_idx * 4
        start_b = b_block_idx * 4
        start_c = c_block_idx * 4
        
        u_full[start_a:start_a+4] = u_block
        v_full[start_b:start_b+4] = v_block
        w_full[start_c:start_c+4] = w_block
        
        return u_full, v_full, w_full

    # The 7 steps of Strassen 4x4:
    # Each step involves a specific A block combination and B block combination.
    # And the result contributes to specific C blocks.
    
    # Step 1: P1 = (A00+A11)(B00+B11) -> C00, C11
    # A blocks: 00, 11. B blocks: 00, 11.
    # C blocks: 00, 11.
    
    # We need to sum the contributions.
    # For each of the 7 terms in the 2x2 decomposition:
    # The u vector is the sum of the u vectors for A00 and A11?
    # No, the Strassen algorithm computes (A00+A11) * (B00+B11).
    # This is a single matrix product.
    # The u vector for this product is the concatenation of the u vectors for A00 and A11?
    # No, A00 and A11 are separate blocks.
    # The product is (A00+A11) * (B00+B11).
    # Let X = A00+A11. Y = B00+B11.
    # X is a 2x2 matrix formed by summing two 2x2 blocks?
    # No, A00 and A11 are 2x2 blocks of the 4x4 matrix.
    # X is a 2x2 matrix where X_ij = A00_ij + A11_ij.
    # So the u vector for X is the sum of the u vectors for A00 and A11?
    # If we treat A00 and A11 as separate inputs, we can't just sum them in the tensor decomposition easily unless we allow linear combinations in the vectors.
    # Yes, we can. The vectors u, v, w can have any real values.
    # So for Step 1, the u vector has non-zero entries in the A00 and A11 blocks.
    # Specifically, u_A00 = u_2x2, u_A11 = u_2x2.
    # So u_full has u_2x2 in block 0 and u_2x2 in block 3.
    
    # Similarly for v.
    # And w has w_2x2 in block 0 and w_2x2 in block 3.
    
    # Let's define the 7 steps with their block indices.
    # (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of (block_idx, sign)
    # b_blocks: list of (block_idx, sign)
    # c_blocks: list of (block_idx, sign)
    
    steps_4x4 = [
        # P1
        {
            'a': [(0, 1), (3, 1)],
            'b': [(0, 1), (3, 1)],
            'c': [(0, 1), (3, 1)]
        },
        # P2
        {
            'a': [(2, 1), (3, 1)],
            'b': [(0, 1)],
            'c': [(2, 1)]
        },
        # P3
        {
            'a': [(0, 1)],
            'b': [(2, 1), (3, -1)],
            'c': [(1, 1)]
        },
        # P4
        {
            'a': [(3, 1)],
            'b': [(1, 1), (0, -1)],
            'c': [(0, 1), (3, 1)]
        },
        # P5
        {
            'a': [(0, 1), (1, 1)],
            'b': [(3, 1)],
            'c': [(0, -1), (1, 1)]
        },
        # P6
        {
            'a': [(2, 1), (0, -1)],
            'b': [(0, 1), (1, 1)],
            'c': [(3, 1)]
        },
        # P7
        {
            'a': [(1, 1), (3, -1)],
            'b': [(2, 1), (3, 1)],
            'c': [(0, 1)]
        }
    ]
    
    col = 0
    for step in steps_4x4:
        for u2, v2, w2 in terms_2x2:
            u2 = np.array(u2)
            v2 = np.array(v2)
            w2 = np.array(w2)
            
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            w_full = np.zeros(16)
            
            for b_idx, sign in step['a']:
                start = b_idx * 4
                u_full[start:start+4] += sign * u2
            
            for b_idx, sign in step['b']:
                start = b_idx * 4
                v_full[start:start+4] += sign * v2
            
            for b_idx, sign in step['c']:
                start = b_idx * 4
                w_full[start:start+4] += sign * w2
            
            U[:, col] = u_full
            V[:, col] = v_full
            W[:, col] = w_full
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Part 2: Schoolbook 4x4x1 (rank 16)
    # Computes A * B[:, 4]
    # B has 5 columns. Index 4 is the 5th column.
    # V rows correspond to (j, k). k=4 is the 5th column.
    # V index = j * 5 + 4.
    # W rows correspond to (k, i). k=4 is the 5th column.
    # W index = 4 * 4 + i.
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
