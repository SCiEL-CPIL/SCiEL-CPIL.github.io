import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 1 for 4x4x1
    rank = 50
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen 4x4x4 (first 4 columns of B)
    # We will fill columns 0 to 48
    
    # Define the 7 Strassen terms for 2x2x2
    # u, v, w are vectors of length 4 (flattened 2x2)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_2x2 = [
        # 1: (A00+A11)(B00+B11) -> C00+C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # 2: (A10+A11)B00 -> C10
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # 3: A00(B10-B11) -> C01
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # 4: A11(B01-B00) -> C10+C11
        ([0, 0, 0, 1], [0, 1, -1, 0], [0, 0, 1, 1]),
        # 5: (A00+A01)B11 -> C00+C01
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        # 6: (A10-A00)(B00+B01) -> C11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # 7: (A01-A11)(B10+B11) -> C00+C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    # Define the 7 Strassen terms for 4x4x4 at block level
    # Each term specifies which 2x2 blocks are involved and their coefficients
    # Block indices: 00, 01, 10, 11
    
    # We need to map block indices to flattened 4x4 indices
    # Block 00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # Block 01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # Block 10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # Block 11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    block_indices_A = {
        0: [0, 1, 4, 5],   # 00
        1: [2, 3, 6, 7],   # 01
        2: [8, 9, 12, 13], # 10
        3: [10, 11, 14, 15] # 11
    }
    
    # For B (4x4), indices are j*4+k
    # Block 00: j=0,1 k=0,1 -> 0,1,4,5
    # Block 01: j=0,1 k=2,3 -> 2,3,6,7
    # Block 10: j=2,3 k=0,1 -> 8,9,12,13
    # Block 11: j=2,3 k=2,3 -> 10,11,14,15
    block_indices_B = block_indices_A
    
    # For C (4x4), indices are k*4+i
    # Block 00: k=0,1 i=0,1 -> 0,1,4,5
    # Block 01: k=0,1 i=2,3 -> 2,3,6,7
    # Block 10: k=2,3 i=0,1 -> 8,9,12,13
    # Block 11: k=2,3 i=2,3 -> 10,11,14,15
    block_indices_C = block_indices_A
    
    # The 7 block-level Strassen terms for 4x4
    # Format: (list of (block_idx, coeff) for A, list for B, list for C)
    
    block_strassen = [
        # 1: (A00+A11)(B00+B11) -> C00+C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # 2: (A10+A11)B00 -> C10
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # 3: A00(B10-B11) -> C01
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),
        # 4: A11(B01-B00) -> C10+C11
        ([(3, 1)], [(1, 1), (0, -1)], [(2, 1), (3, 1)]),
        # 5: (A00+A01)B11 -> C00+C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # 6: (A10-A00)(B00+B01) -> C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # 7: (A01-A11)(B10+B11) -> C00+C11
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
    ]
    
    col = 0
    for block_term in block_strassen:
        a_blocks, b_blocks, c_blocks = block_term
        
        # For each of the 7 element-level Strassen terms
        for elem_u, elem_v, elem_w in strassen_2x2:
            # Build U, V, W vectors for this term
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            # Expand A blocks
            for block_idx, coeff in a_blocks:
                idxs = block_indices_A[block_idx]
                # The element-level Strassen term applies to the 2x2 block
                # So we add coeff * elem_u[i] to the corresponding indices
                for i in range(4):
                    u_vec[idxs[i]] += coeff * elem_u[i]
            
            # Expand B blocks
            for block_idx, coeff in b_blocks:
                idxs = block_indices_B[block_idx]
                for i in range(4):
                    v_vec[idxs[i]] += coeff * elem_v[i]
            
            # Expand C blocks
            for block_idx, coeff in c_blocks:
                idxs = block_indices_C[block_idx]
                for i in range(4):
                    w_vec[idxs[i]] += coeff * elem_w[i]
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
    # Part 2: Rank 1 for 4x4x1 (last column of B, k=4)
    # U: all 1s
    # V: 1s at 5j+4 for j=0..3 -> 4, 9, 14, 19
    # W: 1s at 16+i for i=0..3 -> 16, 17, 18, 19
    
    U[:, 49] = 1.0
    V[4, 49] = 1.0
    V[9, 49] = 1.0
    V[14, 49] = 1.0
    V[19, 49] = 1.0
    W[16, 49] = 1.0
    W[17, 49] = 1.0
    W[18, 49] = 1.0
    W[19, 49] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
