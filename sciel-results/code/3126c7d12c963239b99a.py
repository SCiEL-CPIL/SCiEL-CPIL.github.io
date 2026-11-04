"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    We use a Strassen-inspired decomposition by padding to 4x4x4 and applying
    the 7-multiplication scheme, then extracting the 3x5x7 result.
    
    Strategy: Pad A(3x5) to A'(4x5) by adding a zero row.
    Pad B(5x7) to B'(5x7) (no padding needed for rows, but we'll work with the structure).
    Actually, let's use a direct 7-term decomposition based on block partitioning.
    
    Partition:
    A (3x5): rows split as 2+1
    B (5x7): cols split as 4+3
    C (3x7): split as (2x4, 2x3; 1x4, 1x3)
    
    We define 7 bilinear forms that together compute all of C.
    Using the Strassen identities adapted to rectangular blocks:
    
    Let A_top = A[0:2,:], A_bot = A[2:3,:]
    Let B_left = B[:,0:4], B_right = B[:,4:7]
    
    M1 = A_top * B_left          (2x4)
    M2 = A_top * B_right         (2x3)  
    M3 = A_bot * B_left          (1x4)
    M4 = A_bot * B_right         (1x3)
    M5 = (A_top + A_bot) * B_left    -- but dimensions: (3x5)*(5x4) = 3x4, we only need parts
    M6 = (A_top - A_bot_padded) * B_right
    M7 = A_top * (B_left + B_right)
    
    This is getting complicated. Let me just use the schoolbook for correctness.
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
