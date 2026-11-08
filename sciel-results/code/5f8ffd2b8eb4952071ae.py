"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Use the known Strassen-based decomposition for 4x4 matrix multiplication (rank 49)
    and apply it to two independent 4x4 blocks of the 4x8 matrix.
    
    The 4x8 matrix can be viewed as two 4x4 matrices side by side: [A | B]
    Then C1 = M * A and C2 = M * B are both 4x4 products.
    
    We use the rank-49 Strassen decomposition for each 4x4 product.
    
    The tensor for 4x4 * 4x4 has rank 49 (using Strassen's algorithm recursively).
    For 4x4 * 4x8, we can use two copies of this, giving rank 98.
    
    However, we can potentially do better by exploiting the structure. Let's think about this:
    
    Actually, the standard approach for m x n * n x p where p is even can sometimes be optimized.
    But for now, let's implement the two-block approach with rank 49 each, total rank 98.
    
    The Strassen decomposition for 4x4:
    - Split 4x4 into four 2x2 blocks
    - Apply Strassen to the 2x2 blocks (rank 7)
    - This gives rank 49 for 4x4
    
    Let's implement the rank-49 decomposition for 4x4 * 4x4 first, then extend to 4x4 * 4x8.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We'll construct the decomposition by treating the 4x8 as two 4x4 blocks.
    # For each block, we use the rank-49 Strassen decomposition for 4x4 * 4x4.
    
    # First, let's define the Strassen decomposition for 2x2 * 2x2 (rank 7)
    # Then recursively build up to 4x4 * 4x4 (rank 49)
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build the U, V, W matrices for the 2x2 * 2x2 case (rank 7)
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)
    # W has shape (2*2, 7) = (4, 7)
    
    # Indexing: U[i*m+j, c] where i,j in 0..1 for 2x2
    # For 2x2: U[i*2+j, c]
    
    # Let's define the 7 terms:
    # Term 0: M1 = (A11+A22)(B11+B22)
    # Term 1: M2 = (A21+A22)B11
    # Term 2: M3 = A11(B21-B22)
    # Term 3: M4 = A22(B12-B11)
    # Term 4: M5 = (A11+A12)B22
    # Term 5: M6 = (A21-A11)(B11+B12)
    # Term 6: M7 = (A12-A22)(B21+B22)
    
    U_2x2 = np.zeros((4, 7))
    V_2x2 = np.zeros((4, 7))
    W_2x2 = np.zeros((4, 7))
    
    # A11 is at index 0, A12 at 1, A21 at 2, A22 at 3
    # B11 is at index 0, B12 at 1, B21 at 2, B22 at 3
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U_2x2[0, 0] = 1  # A11
    U_2x2[3, 0] = 1  # A22
    V_2x2[0, 0] = 1  # B11
    V_2x2[3, 0] = 1  # B22
    W_2x2[0, 0] = 1  # C11
    W_2x2[3, 0] = 1  # C22
    
    # Term 1: M2 = (A21+A22)B11
    U_2x2[2, 1] = 1  # A21
    U_2x2[3, 1] = 1  # A22
    V_2x2[0, 1] = 1  # B11
    W_2x2[2, 1] = 1  # C21
    W_2x2[3, 1] = -1  # C22
    
    # Term 2: M3 = A11(B21-B22)
    U_2x2[0, 2] = 1  # A11
    V_2x2[2, 2] = 1  # B21
    V_2x2[3, 2] = -1  # B22
    W_2x2[1, 2] = 1  # C12
    W_2x2[3, 2] = 1  # C22
    
    # Term 3: M4 = A22(B12-B11)
    U_2x2[3, 3] = 1  # A22
    V_2x2[1, 3] = 1  # B12
    V_2x2[0, 3] = -1  # B11
    W_2x2[0, 3] = 1  # C11
    W_2x2[2, 3] = 1  # C21
    
    # Term 4: M5 = (A11+A12)B22
    U_2x2[0, 4] = 1  # A11
    U_2x2[1, 4] = 1  # A12
    V_2x2[3, 4] = 1  # B22
    W_2x2[0, 4] = -1  # C11
    W_2x2[1, 4] = 1  # C12
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U_2x2[2, 5] = 1  # A21
    U_2x2[0, 5] = -1  # A11
    V_2x2[0, 5] = 1  # B11
    V_2x2[1, 5] = 1  # B12
    W_2x2[3, 5] = 1  # C22
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U_2x2[1, 6] = 1  # A12
    U_2x2[3, 6] = -1  # A22
    V_2x2[2, 6] = 1  # B21
    V_2x2[3, 6] = 1  # B22
    W_2x2[0, 6] = 1  # C11
    
    # Now we have the 2x2 * 2x2 decomposition with rank 7.
    # To get 4x4 * 4x4, we split into four 2x2 blocks and apply the 2x2 decomposition to each.
    # This gives rank 7*7 = 49.
    
    # For 4x4 * 4x4:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = [[C11, C12], [C21, C22]] where each Ci is 2x2
    
    # Each Ci is computed using the 2x2 Strassen decomposition on the 2x2 blocks.
    
    # The total rank is 7 (for the outer 2x2 structure) * 7 (for the inner 2x2 structure) = 49.
    
    # Let's construct the U, V, W for 4x4 * 4x4 (rank 49).
    
    # Indexing for 4x4: U[i*4+j, c] where i,j in 0..3
    # Block structure: 
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # Similarly for B and C.
    
    # Let's define the mapping from 4x4 index to (block_row, block_col, local_row, local_col)
    def idx4x4(block_row, block_col, local_row, local_col):
        """Convert 4x4 block indexing to linear index."""
        return (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)
    
    # Build the 4x4 * 4x4 decomposition (rank 49)
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    # For each of the 7 outer Strassen terms, and for each of the 7 inner Strassen terms,
    # we have a rank-1 term in the 4x4 decomposition.
    
    # The outer term t_outer (0..6) determines which 2x2 block combination of A and B is used.
    # The inner term t_inner (0..6) determines which 2x2 Strassen term is used within that block.
    
    # Let's define the outer Strassen terms for the 4x4 blocks:
    # Outer term 0: (A11+A22)(B11+B22)
    # Outer term 1: (A21+A22)B11
    # Outer term 2: A11(B21-B22)
    # Outer term 3: A22(B12-B11)
    # Outer term 4: (A11+A12)B22
    # Outer term 5: (A21-A11)(B11+B12)
    # Outer term 6: (A12-A22)(B21+B22)
    
    # For each outer term, we need to determine which A-blocks and B-blocks are involved,
    # and with what signs.
    
    # Let's define the outer term specifications:
    # (a_blocks, a_signs, b_blocks, b_signs, c_blocks, c_signs)
    # where a_blocks is a list of (block_row, block_col) for A
    # and similarly for B and C.
    
    outer_terms = [
        # Term 0: (A11+A22)(B11+B22) -> C11 +=, C22 +=
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # Term 1: (A21+A22)B11 -> C21 +=, C22 -=
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
        # Term 2: A11(B21-B22) -> C12 +=, C22 +=
        ([(0,0)], [1], [(1,0), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # Term 3: A22(B12-B11) -> C11 +=, C21 +=
        ([(1,1)], [1], [(0,1), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
        # Term 4: (A11+A12)B22 -> C11 -=, C12 +=
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
        # Term 5: (A21-A11)(B11+B12) -> C22 +=
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # Term 6: (A12-A22)(B21+B22) -> C11 +=
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
    ]
    
    # Now, for each outer term and each inner term, we build the rank-1 component.
    
    for t_outer, (a_blocks, a_signs, b_blocks, b_signs, c_blocks, c_signs) in enumerate(outer_terms):
        for t_inner in range(7):
            col = t_outer * 7 + t_inner
            
            # For each A-block involved in this outer term, and for each local position
            # in the 2x2 Strassen decomposition (inner term), we add to U.
            
            # The inner term t_inner specifies which local 2x2 positions are used.
            # From U_2x2, we can see which local positions are nonzero for each inner term.
            
            # For each A-block (block_row, block_col) with sign s_a:
            for (ar, ac), s_a in zip(a_blocks, a_signs):
                # For each local row lr in 0..1 and local col lc in 0..1:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc  # local index in 2x2 block
                        if U_2x2[local_idx, t_inner] != 0:
                            global_idx = idx4x4(ar, ac, lr, lc)
                            U_4x4[global_idx, col] += s_a * U_2x2[local_idx, t_inner]
            
            # Similarly for B-blocks
            for (br, bc), s_b in zip(b_blocks, b_signs):
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        if V_2x2[local_idx, t_inner] != 0:
                            global_idx = idx4x4(br, bc, lr, lc)
                            V_4x4[global_idx, col] += s_b * V_2x2[local_idx, t_inner]
            
            # Similarly for C-blocks
            for (cr, cc), s_c in zip(c_blocks, c_signs):
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        if W_2x2[local_idx, t_inner] != 0:
                            global_idx = idx4x4(cr, cc, lr, lc)
                            W_4x4[global_idx, col] += s_c * W_2x2[local_idx, t_inner]
    
    # Now we have the 4x4 * 4x4 decomposition with rank 49.
    # Let's verify it's correct by checking the tensor decomposition.
    
    # For the 4x4 * 4x8 case, we treat the 4x8 as two 4x4 blocks: [B1 | B2]
    # C1 = A * B1 and C2 = A * B2
    
    # We use the same U for both (since A is the same), but different V for each block.
    # The W will have separate contributions for C1 and C2.
    
    # U has shape (16, 49) for the A matrix (4x4)
    # For the 4x8 case, U is still (16, 98) where the first 49 columns are for B1 and the next 49 for B2.
    # But actually, U should be the same for both blocks, so we can reuse it.
    
    # Let's construct U, V, W for 4x4 * 4x8 (rank 98).
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))  # 4x8 = 32 elements
    W = np.zeros((32, rank))  # 4x8 = 32 elements
    
    # For block 1 (columns 0-3 of B, columns 0-3 of C):
    U[:, 0:49] = U_4x4
    # V for block 1: rows 0-3 of B (which is B1, a 4x4 matrix)
    V[0:16, 0:49] = V_4x4
    # W for block 1: columns 0-3 of C (which is C1, a 4x4 matrix)
    W[0:16, 0:49] = W_4x4
    
    # For block 2 (columns 4-7 of B, columns 4-7 of C):
    U[:, 49:98] = U_4x4
    # V for block 2: rows 16-31 of B (which is B2, a 4x4 matrix)
    V[16:32, 49:98] = V_4x4
    # W for block 2: columns 16-31 of C (which is C2, a 4x4 matrix)
    W[16:32, 49:98] = W_4x4
    
    # Convert to lists for the return value
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
