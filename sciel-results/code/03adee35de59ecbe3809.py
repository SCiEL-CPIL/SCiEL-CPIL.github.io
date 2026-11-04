"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication
    using Strassen's algorithm with padding.
    
    We pad A (3x5) to 4x8, B (5x7) to 8x8, and C (3x7) to 4x8.
    Then apply Strassen's 7-multiplication scheme for 2x2 blocks.
    
    However, direct Strassen on 4x8 x 8x8 gives rank 7*(2*4*4) = 224, which is worse.
    
    Instead, we use a more careful approach: we recognize that the actual nonzero entries
    are only in the first 3 rows, 5 columns of A; first 5 rows, 7 cols of B; and first
    3 rows, 7 cols of C. We can use Strassen on a 4x6 x 6x8 padded problem:
    
    Pad A to 4x6, B to 6x8, C to 4x8.
    Strassen: 7 multiplications of 2x3 x 3x4 -> 2x4, each rank 24, total 168. Still worse.
    
    For this specific small case, the schoolbook rank of 105 is hard to beat with simple
    Strassen padding. We return the schoolbook decomposition.
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
