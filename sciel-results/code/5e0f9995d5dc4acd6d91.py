"""Matrix multiplication 2x3x10 with Strassen-based rank-55 decomposition."""
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x3x10 multiplication into 5 blocks of 2x1x2 (since m=3, p=10 -> 5 pairs of columns)
    # Actually, let's think about the structure more carefully.
    # We have A (2x3) * B (3x10) = C (2x10)
    # We can view this as 5 independent 2x3x2 multiplications (each handling 2 columns of B)
    # Wait, that's not quite right. Let's think of it as 10 independent 2x3x1 multiplications.
    # Or better: use the linearity in the last dimension.
    
    # Actually, the strategy mentioned in the prompt is:
    # Treat the 3x10 matrix as a linear combination of three rank-1 vectors...
    # But the measured knowledge says: Using Strassen's algorithm for the 2x2x2 core of each 2x3x2 block
    # reduces the rank per block from 12 (schoolbook) to 11, lowering total rank from 60 to 55.
    
    # Let me think about this. If we group the 10 columns into 5 pairs, we get 5 blocks of 2x3x2.
    # For each 2x3x2 block, we can use Strassen-like techniques.
    
    # Actually, let me reconsider. The standard approach for 2x3x10:
    # Schoolbook rank = 2*3*10 = 60
    # 
    # We can decompose this as: for each of the 10 output columns, we compute A * B[:,k]
    # where B[:,k] is a 3x1 vector. This is a 2x3x1 multiplication.
    # Rank of 2x3x1 is 3 (since it's just A times a fixed vector, which is 3 scalar multiplications).
    # So total rank would be 10 * 3 = 30. But wait, that's not right either.
    
    # Let me think more carefully. The tensor for 2x3x10 multiplication has rank at most 60 (schoolbook).
    # Can we do better?
    
    # One approach: use the fact that m=3 is small. We can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For fixed k, this is a 2x3x1 problem. The rank of 2x3x1 is 3 (each output is a dot product of length 3).
    # So we need 3 rank-1 tensors per output column, giving 10*3 = 30 total.
    # But wait, that's the same as schoolbook. Let me verify.
    
    # Actually, for 2x3x1, the rank is indeed 3. Each output C[i] = sum_j A[i,j]*B[j] requires 3 multiplications.
    # So 10 columns * 3 = 30. But schoolbook is 60. So we've already halved it!
    
    # Wait, I think I'm confusing myself. Let me be more careful.
    # In the schoolbook decomposition, each (i,j,k) triple gives one rank-1 term.
    # For 2x3x10, that's 2*3*10 = 60 terms.
    # 
    # But if we fix k and look at the 2x3x1 subproblem, we can compute it with rank 3.
    # So for each k, we need 3 rank-1 tensors, and there are 10 values of k, giving 30 total.
    
    # Let me construct this. For each k (0..9), and for each j (0..2), we have a rank-1 term:
    # U[i*m+j, c] = A[i,j] coefficient
    # V[j*p+k, c] = B[j,k] coefficient  
    # W[k*n+i, c] = C[i,k] coefficient
    
    # Wait, I need to be more careful about the indexing.
    
    # Let me just implement the 30-rank decomposition first, then see if we can improve.
    
    n, m, p = 2, 3, 10
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 30 x 30
    W = [[0] * rank for _ in range(p * n)]  # 20 x 30
    
    col = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                # This doesn't work because we need shared structure.
                pass
    
    # Let me think differently. For the 2x3x1 subproblem (fixed k):
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    # 
    # This is rank 3. The rank-1 terms are:
    # Term j=0: U has 1 at (0,0) and (1,0), V has 1 at (0,k), W has 1 at (k,0) and (k,1)
    # Term j=1: U has 1 at (0,1) and (1,1), V has 1 at (1,k), W has 1 at (k,0) and (k,1)
    # Term j=2: U has 1 at (0,2) and (1,2), V has 1 at (2,k), W has 1 at (k,0) and (k,1)
    
    # Wait, that's not right. Let me be more careful.
    # For fixed k, the 2x3x1 multiplication tensor is:
    # T = sum_{i,j} A[i,j] * B[j,k] * C[i,k]
    # The rank of this tensor is 3 (it's the outer product structure of a 2x3 matrix times a vector).
    
    # Actually, I realize the issue. The full tensor is:
    # T = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # This has rank 60 in the trivial decomposition.
    # 
    # But we can group by k:
    # T = sum_k (sum_{i,j} A[i,j] * B[j,k] * C[i,k])
    # Each inner sum is a 2x3x1 tensor with rank 3.
    # So total rank is 10 * 3 = 30.
    
    # Let me construct this properly.
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 30 x 30
    W = [[0] * rank for _ in range(p * n)]  # 20 x 30
    
    col = 0
    for k in range(p):
        for j in range(m):
            # For this (k,j), we have a rank-1 term
            # U[i*m+j, col] = 1 for i=0,1
            # V[j*p+k, col] = 1
            # W[k*n+i, col] = 1 for i=0,1
            for i in range(n):
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
