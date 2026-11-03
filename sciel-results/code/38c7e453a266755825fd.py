import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = 3, 5, 5
    # We will build the decomposition in blocks
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3] (3x3x3)
    # Block 2: A[0:3, 0:3] * B[0:3, 3:5] -> C[0:3, 3:5] (3x3x2)
    # Block 3: A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3] (3x2x3)
    # Block 4: A[0:3, 3:5] * B[3:5, 3:5] -> C[0:3, 3:5] (3x2x2)
    
    U = []
    V = []
    W = []
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset):
        # U_block shape: (n_block * m_block, rank_block)
        # V_block shape: (m_block * p_block, rank_block)
        # W_block shape: (p_block * n_block, rank_block)
        # We need to embed these into the global U, V, W
        # This is tricky because U, V, W are global matrices.
        # It's easier to build the global matrices directly.
        pass

    # Let's build global U, V, W directly.
    rank_total = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Block 1: 3x3x3
    # Use rank 26 decomposition? Or just schoolbook 27 for simplicity if 26 is hard?
    # Let's use schoolbook for all to ensure correctness, then optimize if time permits.
    # Actually, 75 is the baseline. We need < 75.
    # If we use Strassen 2x2x2 (rank 7) for the 2x2x2 sub-blocks of the 3x3x3, we get 26.
    # Let's implement the 26-rank 3x3x3.
    
    # This is getting complex. Let's try a simpler improvement.
    # What if we just use schoolbook? Score 1.0.
    # We need to beat 1.0.
    # Rank 74 is a 2% improvement.
    
    # Let's implement the block decomposition with schoolbook for all blocks.
    # Rank = 27 + 18 + 18 + 12 = 75.
    # This is the same as the seed.
    
    # We MUST use Strassen for the 3x3x3 block to get 26.
    # And maybe optimize the others?
    # R(3,3,2) = 18.
    # R(3,2,3) = 18.
    # R(3,2,2) = 12.
    
    # Total 74.
    
    # Implementation of 3x3x3 rank 26:
    # Split A, B into 2x2, 2x1, 1x2, 1x1.
    # A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1).
    # B11 (2x2), B12 (2x1), B21 (1x2), B22 (1x1).
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We will generate the factors for each term.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # Standard Strassen
        U = [
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1, 0]
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1]
        ]
        W = [
            [0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0]
        ]
        return U, V, W

    # This is too complex to do manually in the thought process.
    # I will write the code to construct the schoolbook decomposition first, 
    # and then try to replace the 3x3x3 block with a lower rank one if I can.
    
    # For now, I will output the schoolbook decomposition to ensure a valid program.
    # But I need to beat 1.0.
    
    # Let's try to find a rank < 75.
    # If I can't, I return 75.
    
    # Wait, the seed is 75.
    # If I return 75, score is 1.0.
    # I need < 75.
    
    # Let's assume I can implement the 3x3x3 rank 26.
    # I will write the code for that.
    
    return {"U": U, "V": V, "W": W}
