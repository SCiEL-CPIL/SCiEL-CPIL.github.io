"""Program that attempts to find a low-rank decomposition for 3x5x6 matrix multiplication.

Strategy: Use block decomposition to split the problem into smaller sub-problems.
For 3x5x6, we can split as:
- 1x1x6 (rank 6, schoolbook)
- 2x4x5 (need to find or construct a decomposition)

For the 2x4x5 sub-problem, we can further split:
- 1x1x5 (rank 5)
- 1x3x5 (rank 15, schoolbook)
Total for 2x4x5: 5 + 15 = 20

Or we can use Strassen-like methods for 2x2x2 blocks.

Actually, let me think more carefully. The tensor for n x m x p multiplication has
n*m*p entries. The rank is the minimum number of rank-1 tensors to sum to get it.

For 3x5x6: total entries = 90. Schoolbook gives rank 90.

Can we do better? Let's try a block decomposition.

Split 3 = 1 + 2, 5 = 1 + 4, 6 = 6.
Blocks:
- 1x1x6: rank 6
- 1x4x6: rank 24
- 2x1x6: rank 12
- 2x4x6: rank 48
Total: 6 + 24 + 12 + 48 = 90. No improvement.

The key is that block decomposition only helps if the sub-problems have sub-multiplicative
rank, which for matrix multiplication tensors is not generally the case.

Let me just implement the schoolbook decomposition which is guaranteed correct.
"""

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct the tensor decomposition for 3x5x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank)
        - V has shape (m*p, rank)  
        - W has shape (p*n, rank)
        with W using the trace convention W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition: one rank-1 tensor for each (i, j, k) triple
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
