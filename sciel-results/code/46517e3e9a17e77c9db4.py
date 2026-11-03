import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on splitting 4x4 into 3x3 and 1x1 blocks.
    # Rank(3x3x3) = 23 (Laderman)
    # Rank(3x1x3) = 9
    # Rank(3x3x1) = 9
    # Rank(1x3x3) = 9
    # Rank(1x3x1) = 3
    # Rank(1x1x3) = 3
    # Rank(1x1x1) = 1
    # Total for 4x4x4 part = 23 + 9 + 9 + 9 + 3 + 3 + 1 = 57
    
    # For the 5th column (4x4x1), rank is 16.
    # Total rank = 57 + 16 = 73.
    
    rank_total = 73
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook decomposition for n x m x p block at offset i_off, j_off, k_off
    def add_schoolbook(n_block, m_block, p_block, i_off, j_off, k_off, U, V, W, col):
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[i_off + i][col] = 1
                    V[j_off + j * p_block + k][col] = 1
                    # W index is k * n + i
                    W[k_off + k][col] = 1
                    col += 1
        return col

    # 1. A11 (3x3) * B11 (3x3) -> C11 (3x3) part
    # Laderman 3x3 decomposition
    # Coefficients for Laderman's algorithm for 3x3
    # U, V, W matrices for 3x3x3
    # We need to embed these into the 4x4 grid.
    
    # Laderman's 23 terms
    # Defined on 0..2 indices
    laderman_U = [
        [1,0,0, 0,1,0, 0,0,1, 1,1,0, 0,0,0, 0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        # This is getting complicated to hardcode Laderman exactly.
        # Let's stick to the Schoolbook for 3x3 to be safe? Rank 27.
        # Then total: 27 + 9 + 9 + 9 + 3 + 3 + 1 = 61.
        # 61 + 16 = 77. Still < 80.
    ]
    
    # Let's use Schoolbook for all 3x3 blocks to ensure correctness.
    # Rank(3x3x3) = 27.
    # Total 4x4x4 part: 27 + 9 + 9 + 9 + 3 + 3 + 1 = 61.
    # Total 4x4x5 part: 61 + 16 = 77.
    
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3]
    # U: i in 0..2, j in 0..2
    # V: j in 0..2, k in 0..2
    # W: k in 0..2, i in 0..2
    col = add_schoolbook(3, 3, 3, 0, 0, 0, U, V, W, col)
    
    # Block 2: A[0:3, 3] * B[3, 0:3] -> C[0:3, 0:3]
    # A is 3x1, B is 1x3.
    # i in 0..2, j=3, k in 0..2
    col = add_schoolbook(3, 1, 3, 0, 3, 0, U, V, W, col)
    
    # Block 3: A[0:3, 0:3] * B[0:3, 3] -> C[0:3, 3]
    # A is 3x3, B is 3x1.
    # i in 0..2, j in 0..2, k=3
    col = add_schoolbook(3, 3, 1, 0, 0, 3, U, V, W, col)
    
    # Block 4: A[3, 0:3] * B[0:3, 0:3] -> C[3, 0:3]
    # A is 1x3, B is 3x3.
    # i=3, j in 0..2, k in 0..2
    col = add_schoolbook(1, 3, 3, 3, 0, 0, U, V, W, col)
    
    # Block 5: A[3, 0:3] * B[0:3, 3] -> C[3, 3]
    # A is 1x3, B is 3x1.
    # i=3, j in 0..2, k=3
    col = add_schoolbook(1, 3, 1, 3, 0, 3, U, V, W, col)
    
    # Block 6: A[3, 3] * B[3, 0:3] -> C[3, 0:3]
    # A is 1x1, B is 1x3.
    # i=3, j=3, k in 0..2
    col = add_schoolbook(1, 1, 3, 3, 3, 0, U, V, W, col)
    
    # Block 7: A[3, 3] * B[3, 3] -> C[3, 3]
    # A is 1x1, B is 1x1.
    # i=3, j=3, k=3
    col = add_schoolbook(1, 1, 1, 3, 3, 3, U, V, W, col)
    
    # Block 8: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # A is 4x4, B is 4x1.
    # i in 0..3, j in 0..3, k=4
    col = add_schoolbook(4, 4, 1, 0, 0, 4, U, V, W, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
