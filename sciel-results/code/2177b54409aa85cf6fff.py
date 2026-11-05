"""Matrix multiplication 2x5x6 with structured low-rank completion approach."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We use a block decomposition approach:
    # Split the middle dimension j=0..4 into two blocks:
    #   Block 1: j=0,1 (size 2)
    #   Block 2: j=2,3,4 (size 3)
    #
    # For Block 1 (2x2x6), we use a Strassen-like decomposition for the 2x2x2 
    # sub-problem applied to each pair of output columns, but more efficiently:
    #   For each pair of output columns (k, k+1), we can use Strassen on the 
    #   2x2x2 sub-tensor. But actually, let's think more carefully.
    #
    # Actually, a cleaner approach: For the 2x2x6 sub-problem, we can use 
    # Strassen's algorithm for 2x2x2 block multiplication. But since the 
    # "block size" in the k-direction is 1 (not 2), we need to apply Strassen 
    # differently.
    #
    # Let me reconsider. The 2x2x6 tensor has entries T[i,j,k] for i=0,1; j=0,1; k=0..5.
    # This is equivalent to 6 separate 2x2x1 tensors (one for each k), but they share 
    # the same A and B structure across k.
    #
    # A better approach: Use the fact that for fixed i,j, the values across k are 
    # just B[j,k]. So the tensor is:
    #   T[i,j,k] = A[i,j] * B[j,k]
    #
    # For the 2x2x6 case, we can use a decomposition that shares computation 
    # across k. One known result: the rank of 2x2xn matrix multiplication is 
    # at most 7*floor(n/2) + 4*ceil(n/2) or similar, but let's construct it directly.
    #
    # Actually, let's use a simpler but effective approach:
    # For the 2x2x6 sub-problem, use schoolbook: rank = 2*2*6 = 24.
    # For the 2x3x6 sub-problem, use schoolbook: rank = 2*3*6 = 36.
    # Total: 60. No improvement.
    #
    # To improve, we need to find a better decomposition for at least one block.
    #
    # Key insight: For 2x2x2, Strassen gives rank 7 instead of 8.
    # For 2x2x6, we can group the 6 k-values into 3 pairs of 2, and apply Strassen 
    # to each pair. But Strassen works on 2x2x2 blocks, so if we group k into 
    # pairs, we get 3 independent 2x2x2 sub-problems, each with rank 7, total 21.
    # But wait, that's not quite right because the A and B matrices are shared 
    # across all k values.
    #
    # Let me try a direct construction for 2x2x6 with rank < 24.
    #
    # Actually, let's use a different strategy. Instead of splitting j into {0,1} 
    # and {2,3,4}, let's try to find a global decomposition.
    #
    # For 2x5x6, one effective approach is to use the "border rank" or "decomposition 
    # with shared factors". 
    #
    # Let me try: For each of the 12 output positions (i,k) where i=0,1 and k=0..5, 
    # we need C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k].
    #
    # If we can find a set of bilinear forms that cover all 12 outputs with fewer 
    # than 60 multiplications, we win.
    #
    # Known lower bound: The rank of 2x5x6 is at least max(2*5, 5*6, 2*6) = 30, 
    # but actually the lower bound is higher. The true rank is between 30 and 60.
    #
    # Let me try a construction with rank 58:
    # Use schoolbook for all but 2 terms, and replace 2 terms with a shared factor.
    #
    # Actually, let's just implement a correct schoolbook decomposition for now, 
    # and then try to optimize.
    
    rank = n * m * p  # 60
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
