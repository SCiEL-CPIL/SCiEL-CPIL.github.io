"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    Strategy: Apply Strassen's divide-and-conquer by padding to 4x4 and 4x8 structure.
    We partition A (3x5) as 2x5 + 1x5 (pad to 2x5 + 2x5 = 4x5)
    We partition B (5x7) as 5x4 + 5x3 (pad to 5x4 + 5x4 = 5x8)
    
    Actually, let's use a cleaner approach:
    Pad A to 4x5 (add one zero row)
    Pad B to 5x8 (add one zero column)
    Pad C to 4x8 (add one zero row and one zero column)
    
    Now we have a 4x5 * 5x8 = 4x8 multiplication.
    We can further pad to 4x8 * 8x8 = 4x8 or use block Strassen.
    
    Better approach: Use the fact that we can decompose using Strassen on 2x2 blocks
    where each block is of appropriate size.
    
    Let me use a different strategy. We'll use the following decomposition:
    
    Partition A (3x5) into two parts: A_top (2x5) and A_bot (1x5)
    Partition B (5x7) into two parts: B_left (5x4) and B_right (5x3)
    
    We want to compute:
    C_top_left (2x4) = A_top * B_left
    C_top_right (2x3) = A_top * B_right  
    C_bot_left (1x4) = A_bot * B_left
    C_bot_right (1x3) = A_bot * B_right
    
    Using Strassen-style formulas, we can compute these with 7 products instead of 4.
    But the blocks have different sizes, so we need to be careful.
    
    Let me try a more direct approach: use the known optimal decompositions for smaller
    matrix multiplication problems and combine them.
    
    For now, let me implement a simple but improved decomposition using the following:
    
    We'll use the fact that we can write:
    C = A * B = sum_{j=0}^{4} A[:,j] * B[j,:]
    
    This is the schoolbook with rank 105.
    
    To improve, let's use a block decomposition with Strassen where applicable.
    
    Actually, let me try the following approach:
    
    Use the decomposition for 2x2x2 Strassen and combine it with schoolbook for the remaining parts.
    
    Partition:
    A (3x5) = [A11 (1x1), A12 (1x4); A21 (2x1), A22 (2x4)]
    B (5x7) = [B11 (1x3), B12 (1x4); B21 (4x3), B22 (4x4)]
    
    C (3x7) = [C11 (1x3), C12 (1x4); C21 (2x3), C22 (2x4)]
    
    Where:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    For the 2x2 blocks (A21, A22, B21, B22, C21, C22), we can use Strassen:
    A21 is 2x1, A22 is 2x4, B21 is 4x3, B22 is 4x4
    C21 is 2x3, C22 is 2x4
    
    This is not a standard 2x2 block structure for Strassen since the blocks aren't square.
    
    Let me try yet another approach: use a direct rank-35 construction if possible,
    or fall back to a better block decomposition.
    
    For simplicity and correctness, let me implement a block decomposition that
    uses Strassen where the blocks are square, and schoolbook otherwise.
    
    Partition A (3x5) as:
    - Rows 0-1: 2x5
    - Row 2: 1x5
    
    Partition B (5x7) as:
    - Cols 0-3: 5x4
    - Cols 4-6: 5x3
    
    For the 2x5 * 5x4 = 2x4 multiplication, use schoolbook: rank = 2*5*4 = 40
    For the 2x5 * 5x3 = 2x3 multiplication, use schoolbook: rank = 2*5*3 = 30
    For the 1x5 * 5x4 = 1x4 multiplication, use schoolbook: rank = 1*5*4 = 20
    For the 1x5 * 5x3 = 1x3 multiplication, use schoolbook: rank = 1*5*3 = 15
    
    Total: 40 + 30 + 20 + 15 = 105
    
    This is the same as before. Let me try to use Strassen on the 2x5 * 5x4 part.
    
    Actually, I realize that for this specific problem, achieving rank 35 requires
    a very specific construction. Let me try to implement a known optimal decomposition.
    
    For now, let me return the schoolbook decomposition to ensure correctness.
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
