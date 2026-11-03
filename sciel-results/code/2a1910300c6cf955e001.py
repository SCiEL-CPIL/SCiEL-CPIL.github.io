import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose 4x4x8 into two 4x4x4 multiplications.
    # Each 4x4x4 uses Strassen's algorithm with rank-7 2x2x2 kernels.
    # Rank per 4x4x4 is 49. Total rank is 98.
    
    # Rank-7 decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    U_2 = [
        [0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, -1, 0],
        [1, 1, -1, 1, 0, 0, 1]
    ]
    V_2 = [
        [1, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, -1, 1, -1],
        [-1, 1, 0, 0, 1, 1, 0]
    ]
    W_2 = [
        [0, 0, 1, 0, -1, 0, 1],
        [1, 0, -1, 1, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [-1, 1, -1, 0, 0, 0, 0]
    ]
    
    # Strassen terms for 4x4x4
    # Each term is a multiplication of two 2x2 blocks.
    # We need to map the local 2x2 block indices to global 4x4 indices.
    # Blocks:
    # A: 0:00-11, 1:02-13, 2:20-31, 3:22-33
    # B: 0:00-11, 1:02-13, 2:20-31, 3:22-33
    # C: 0:00-11, 1:02-13, 2:20-31, 3:22-33
    
    # Term definitions: (A_blocks, B_blocks, C_blocks, signs)
    # A_blocks: list of (block_id, coeff)
    # B_blocks: list of (block_id, coeff)
    # C_blocks: list of (block_id, coeff)
    
    terms = [
        # M1 = A22 * (B11 - B22)
        {
            'A': [(3, 1)],
            'B': [(0, 1), (3, -1)],
            'C': [(2, 1), (3, -1)]
        },
        # M2 = (A11 + A22) * B22
        {
            'A': [(0, 1), (3, 1)],
            'B': [(3, 1)],
            'C': [(1, 1), (3, 1)]
        },
        # M3 = (A11 - A22) * (B11 + B12)
        {
            'A': [(0, 1), (3, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(0, 1), (2, -1), (1, 1), (3, -1)]
        },
        # M4 = (A21 + A22) * B11
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1)]
        },
        # M5 = (A11 + A12) * (B22 - B21)
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1), (2, -1)],
            'C': [(0, -1)]
        },
        # M6 = (A12 - A21) * (B11 + B22)
        {
            'A': [(1, 1), (2, -1)],
            'B': [(0, 1), (3, 1)],
            'C': [(2, 1)]
        },
        # M7 = (A11 + A22) * (B12 - B21)
        {
            'A': [(0, 1), (3, 1)],
            'B': [(1, 1), (2, -1)],
            'C': [(0, 1), (2, 1)]
        }
    ]
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to map block_id and local index to global index
    def map_block_to_global(block_id, local_i, local_j, dim):
        # block_id 0: 0-1, 1: 0-1 (cols 2-3), 2: 2-3, 3: 2-3
        # rows: block 0,1 -> 0,1; block 2,3 -> 2,3
        # cols: block 0,2 -> 0,1; block 1,3 -> 2,3
        row_base = (block_id // 2) * 2
        col_base = (block_id % 2) * 2
        return (row_base + local_i) * dim + (col_base + local_j)
        
    # Note: U is n*m, V is m*p, W is p*n
    # n=4, m=4, p=4 for the 4x4x4 part.
    # But our global p is 8.
    # We split B into B1 (4x4) and B2 (4x4).
    # C1 = A B1, C2 = A B2.
    # C1 indices: k in 0..3, i in 0..3. W index k*4+i.
    # C2 indices: k in 4..7, i in 0..3. W index k*4+i.
    
    # We will construct the first 49 columns for C1 and next 49 for C2.
    
    for part in range(2):
        offset = part * 49
        for t_idx in range(7):
            term = terms[t_idx]
            # For each of the 7 rank-1 components of the 2x2x2 decomposition
            for r in range(7):
                col = offset + t_idx * 7 + r
                # U part
                for local_i in range(2):
                    for local_j in range(2):
                        val = U_2[local_i * 2 + local_j][r]
                        if val != 0:
                            # Apply coefficients from term['A']
                            # Wait, U_2 already contains the linear combination?
                            # No, U_2 is the decomposition of the 2x2x2 tensor.
                            # The term specifies which blocks are involved.
                            # The local index in U_2 corresponds to the block's internal indices.
                            # But we need to map the block ID.
                            
                            # This approach is slightly flawed.
                            # U_2[r] is a vector of 4 values for the 2x2 matrix A.
                            # These 4 values correspond to A11, A12, A21, A22 *of the block*.
                            # But in Strassen, we multiply Block A by Block B.
                            # So the "A" in the 2x2x2 decomposition is the Block A.
                            # So U_2 values apply to the elements of Block A.
                            
                            # We need to iterate over the blocks involved in term['A'].
                            pass
                
                # Correct approach:
                # For each block in term['A'], add its contribution.
                # The contribution of a block is U_2[local_idx] * coeff.
                # local_idx is 0,1,2,3 for the 2x2 block.
                
                for block_id, coeff in term['A']:
                    for local_i in range(2):
                        for local_j in range(2):
                            local_idx = local_i * 2 + local_j
                            val = U_2[local_idx][r] * coeff
                            if val != 0:
                                global_idx = map_block_to_global(block_id, local_i, local_j, 4)
                                U[global_idx][col] += val
                                
                for block_id, coeff in term['B']:
                    for local_i in range(2):
                        for local_j in range(2):
                            local_idx = local_i * 2 + local_j
                            val = V_2[local_idx][r] * coeff
                            if val != 0:
                                # B is 4x4. Block mapping is same.
                                global_idx = map_block_to_global(block_id, local_i, local_j, 4)
                                V[global_idx][col] += val
                                
                for block_id, coeff in term['C']:
                    for local_i in range(2):
                        for local_j in range(2):
                            local_idx = local_i * 2 + local_j
                            val = W_2[local_idx][r] * coeff
                            if val != 0:
                                # C is 4x4. W index is k*4+i.
                                # local_i is row i, local_j is col k.
                                # map_block_to_global returns i*4+j.
                                # We need k*4+i.
                                # So swap i and j in the mapping?
                                # map_block_to_global(block_id, local_i, local_j, 4) gives (row)*4 + col.
                                # For W, we need col*4 + row.
                                # So we call map_block_to_global(block_id, local_j, local_i, 4).
                                global_idx = map_block_to_global(block_id, local_j, local_i, 4)
                                # Adjust for part
                                if part == 1:
                                    global_idx += 4 * 4 # Shift by 16?
                                    # W size is p*n = 8*4 = 32.
                                    # Part 0: k in 0..3. Index k*4+i.
                                    # Part 1: k in 4..7. Index k*4+i.
                                    # So add 4*4 = 16 to the index.
                                W[global_idx][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
