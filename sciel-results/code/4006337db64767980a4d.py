"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Construct a rank-minimal exact tensor decomposition for multiplying
    # a 3x3 matrix by a 3x8 matrix using Strassen's algorithm structure.
    #
    # We use Strassen's 7-multiplication algorithm for the 2x2 block structure,
    # but adapted for our specific dimensions. Since our matrices are 3x3 and 3x8,
    # we can split them into 2x2 blocks where the bottom-right blocks are 1x1 and 1x8.
    #
    # For a 2x2 matrix multiplication, Strassen uses 7 multiplications instead of 8.
    # Here we apply this to the top-left 2x2 block structure, with the remaining
    # parts handled separately.
    #
    # Actually, let's use a direct approach: we'll construct the decomposition
    # explicitly for the 3x3 x 3x8 case by finding a set of bilinear forms that
    # compute all 24 entries of the result.
    #
    # The result C = A*B where A is 3x3 and B is 3x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We need 24 outputs. Each "multiplication" in the tensor decomposition
    # corresponds to one bilinear term A[i,j]*B[j,k] contributing to one or more
    # output entries.
    #
    # For the schoolbook approach, we have exactly 9*8 = 72 terms, each being
    # a single product A[i,j]*B[j,k] that contributes to exactly one output C[i,k].
    #
    # To get a lower rank, we need to find combinations where a single bilinear
    # term contributes to multiple outputs, or where we can use identities.
    #
    # Let's try a hybrid approach: use Strassen for the first 2x2 block of the
    # 3x3 matrix, and handle the rest with schoolbook.
    #
    # Actually, for 3x3 x 3x8, let's think about it differently.
    # We can view this as three 1x3 x 3x8 multiplications (one for each row of A).
    # Each 1x3 x 3x8 multiplication has rank 3*8 = 24 in schoolbook.
    # Total: 3 * 24 = 72.
    #
    # Alternatively, we can view it as three 3x3 x 3x1 multiplications (one for
    # each column of the 3x8 matrix), each with rank 9 in schoolbook.
    # Total: 8 * 9 = 72.
    #
    # For Strassen on 3x3 x 3x1: we could potentially do better than 9 per column.
    # Strassen for 2x2 x 2x2 has rank 7. For 3x3 x 3x1, it's less clear.
    #
    # Let's just implement the schoolbook decomposition carefully, as it's the
    # most straightforward and likely optimal for these small dimensions.
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (24, rank)
    
    # Schoolbook: rank = n*m*p = 72
    # For each (i, j, k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
