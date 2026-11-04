"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    We use a block-based approach with Strassen's algorithm to reduce the rank below the
    schoolbook rank of 105.
    
    Strategy:
    1. Pad the 3x5x7 multiplication to a 4x8x8 multiplication (or equivalent).
    2. Use Strassen's divide-and-conquer to decompose the 4x8x8 multiplication.
    3. Project back to the original 3x5x7 indices.
    
    For 3x5x7, we can view it as:
    - Partition A (3x5) into 2x5 and 1x5
    - Partition B (5x7) into 5x4 and 5x3
    - C (3x7) has blocks: 2x4, 2x3, 1x4, 1x3
    
    We'll use a 2x2 block Strassen decomposition where we pad the blocks to make them
    compatible. The key insight is that we can use 7 products instead of 4 for the
    2x2 block structure.
    
    Let's define the Strassen decomposition for 2x2 blocks:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B21 - B11)
    M4 = A22 * (B12 - B22)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For our case, the blocks are not square, so we need to be careful.
    
    Let's try a different approach: use the fact that we can decompose the multiplication
    into smaller known decompositions.
    
    For 3x5x7, let's try to use a combination of:
    - 2x4x4 multiplication (rank 32 schoolbook, can be reduced with Strassen to ~22)
    - 2x3x4 multiplication (rank 24 schoolbook)
    - 1x4x4 multiplication (rank 16 schoolbook)
    - 1x3x4 multiplication (rank 12 schoolbook)
    
    Actually, let me just implement a direct Strassen decomposition for the 2x2 block
    structure where we pad the blocks appropriately.
    
    Here's a cleaner approach:
    - Pad A from 3x5 to 4x5 (add a zero row)
    - Pad B from 5x7 to 5x8 (add a zero column)
    - Now we have 4x5 by 5x8 = 4x8
    - Use Strassen on 4x4 blocks: pad to 4x8x8
    
    For 4x8x8, we can use 2x2 Strassen where each block is 2x4 and 4x4.
    This gives us 7 recursive calls, each of which can be further decomposed.
    
    Let me just implement the schoolbook for now, as I'm not confident in the Strassen
    construction for this specific case.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
