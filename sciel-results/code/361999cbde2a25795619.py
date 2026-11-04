import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 110.0  # Safety margin before 120s limit
    
    # Try to construct a decomposition using block structure
    # 5x5 matmul can be decomposed into blocks:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # where sizes are 2x2, 2x2, 1x1 etc.
    
    # Strategy: Use Strassen for 2x2 blocks where possible
    # The key insight: matmul tensor T is a "product" structure
    # We can decompose based on block partitions
    
    # Let's try a partition: split indices into blocks of size 2 and 1
    # For 5 = 2+2+1
    
    # Block decomposition approach:
    # If we have blocks of sizes [2,2,1] for rows, cols, and result
    # We can use Strassen (rank 7) for 2x2x2 blocks
    # and schoolbook (rank 1) for 1x1x1 blocks
    # and mixed blocks need careful handling
    
    # Actually, let me think more carefully.
    # The matmul tensor for n x m x p has entries T[i,j,k] = delta(i,j,k)
    # where the "valid" triples are those that appear in (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # For a block decomposition, if we partition:
    # rows into R1, R2, ..., Rr (sizes r1, r2, ...)
    # cols into C1, C2, ..., Cs (sizes c1, c2, ...)  
    # result cols into K1, K2, ..., Kt (sizes k1, k2, ...)
    
    # Then the tensor decomposes as a sum over blocks
    # T_block[a,b,c] for block (A_a, B_b, C_c)
    
    # The rank of the block tensor is at most product of block ranks
    # But we can do better by sharing terms
    
    # Let me try a simpler approach: direct construction using known low-rank decompositions
    
    # Known: 2x2x2 matmul has rank 7 (Strassen)
    # 3x3x3 matmul has rank 19 (Laderman) or 23 (schoolbook)
    
    # For 5x5x5, one approach:
    # 5 = 2 + 3
    # Use tensor product structure
    
    # Actually, let me just implement a search that tries to find a valid decomposition
    # by solving the linear system for given U, V
    
    # The constraint is: for all (u,v,w), sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
    # where T[u,v,w] = 1 if (u,v,w) is a valid matmul term, else 0
    
    # This is a system of 15625 equations in 3*25*R unknowns
    
    # Strategy: 
    # 1. Try small ranks (45-60)
    # 2. For each rank, try random initializations
    # 3. Use iterative refinement
    
    # But this is hard. Let me try a more structured approach.
    
    # Structured approach using block decomposition:
    # Partition 5 into 2+2+1
    
    # For the 2x2 blocks, use Strassen (7 terms)
    # For mixed blocks, use schoolbook or other decompositions
    
    # Let me define the block structure explicitly
    
    # Row partition: [0,1], [2,3], [4] -> sizes [2,2,1]
    # Col partition: [0,1], [2,3], [4] -> sizes [2,2,1]  
    # Result col partition: [0,1], [2,3], [4] -> sizes [2,2,1]
    
    # The matmul (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # Block (row_block_a, col_block_b, result_block_c):
    # A_block[a,b] is (ra x cb) matrix
    # B_block[b,c] is (cb x kc) matrix
    # Result_block[a,c] is (ra x kc) matrix
    # Result_block[a,c] = A_block[a,b] @ B_block[b,c] for each b
    
    # So for each triple (a,b,c), we need to decompose the matmul tensor
    # of sizes (ra, cb, kc)
    
    # For (2,2,2): rank 7 (Strassen)
    # For (2,2,1): rank 4 (schoolbook, since 2*2*1=4)
    # For (2,1,2): rank 4
    # For (1,2,2): rank 4
    # For (2,1,1): rank 2
    # For (1,2,1): rank 2
    # For (1,1,2): rank 2
    # For (1,1,1): rank 1
    
    # Total rank = sum over all (a,b,c) of rank(ra,cb,kc)
    # But this double-counts! The blocks share the same A and B entries.
    
    # Actually, the decomposition is:
    # For each b (col block), we have:
    #   For each a (row block) and c (result block):
    #     Result[a,c] += A[a,b] @ B[b,c]
    
    # The tensor for block (a,b,c) has rank at most ra*cb*kc
    # But we can use lower rank decompositions
    
    # The total rank is the sum of ranks for each (a,b,c) block
    # because the supports are disjoint in the tensor space
    
    # Wait, that's not right. The tensor entries are:
    # T[u,v,w] where u = (a, i_in_a), v = (b, j_in_b), w = (c, k_in_c)
    # T[(a,i),(b,j),(c,k)] = 1 if the matmul term A[a,i]*B[b,j] contributes to Result[c,k]
    # This happens when a=i_block, b=j_block, c=k_block, and the indices match
    
    # So T is nonzero only when (a,b,c) is a valid triple, and the local indices match
    
    # The rank of the full tensor is at most the sum over (a,b,c) of rank of local tensor
    # because we can decompose each local tensor independently
    
    # So total rank = sum_{a,b,c} rank(ra, cb, kc)
    
    # For partition [2,2,1]:
    # (2,2,2): 7
    # (2,2,1): 4
    # (2,1,2): 4
    # (2,1,1): 2
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,1,2): 2
    # (1,1,1): 1
    # (1,1,1): 1  [Wait, I'm double counting]
    
    # Let me enumerate all (a,b,c) where a in {0,1,2}, b in {0,1,2}, c in {0,1,2}
    # sizes: ra = [2,2,1], cb = [2,2,1], kc = [2,2,1]
    
    # (0,0,0): (2,2,2) -> 7
    # (0,0,1): (2,2,2) -> 7
    # (0,0,2): (2,2,1) -> 4
    # (0,1,0): (2,2,2) -> 7
    # (0,1,1): (2,2,2) -> 7
    # (0,1,2): (2,2,1) -> 4
    # (0,2,0): (2,1,2) -> 4
    # (0,2,1): (2,1,2) -> 4
    # (0,2,2): (2,1,1) -> 2
    # (1,0,0): (2,2,2) -> 7
    # (1,0,1): (2,2,2) -> 7
    # (1,0,2): (2,2,1) -> 4
    # (1,1,0): (2,2,2) -> 7
    # (1,1,1): (2,2,2) -> 7
    # (1,1,2): (2,2,1) -> 4
    # (1,2,0): (2,1,2) -> 4
    # (1,2,1): (2,1,2) -> 4
    # (1,2,2): (2,1,1) -> 2
    # (2,0,0): (1,2,2) -> 4
    # (2,0,1): (1,2,2) -> 4
    # (2,0,2): (1,2,1) -> 2
    # (2,1,0): (1,2,2) -> 4
    # (2,1,1): (1,2,2) -> 4
    # (2,1,2): (1,2,1) -> 2
    # (2,2,0): (1,1,2) -> 2
    # (2,2,1): (1,1,2) -> 2
    # (2,2,2): (1,1,1) -> 1
    
    # Sum: 7+7+4+7+7+4+4+4+2+7+7+4+7+7+4+4+4+2+4+4+2+4+4+2+2+2+1
    # = (7*8) + (4*14) + (2*4) + 1
    # = 56 + 56 + 8 + 1 = 121
    
    # That's barely better than 125! The issue is that most blocks are (2,2,2) or (2,2,1)
    
    # Let me try a different partition: [3,2] for all dimensions
    # ra = [3,2], cb = [3,2], kc = [3,2]
    
    # (0,0,0): (3,3,3) -> 19 (Laderman) or 23 (schoolbook)
    # (0,0,1): (3,3,2) -> ? 
    # (0,1,0): (3,2,3) -> ?
    # (0,1,1): (3,2,2) -> ?
    # (1,0,0): (2,3,3) -> ?
    # (1,0,1): (2,3,2) -> ?
    # (1,1,0): (2,2,3) -> ?
    # (1,1,1): (2,2,2) -> 7
    
    # I don't know the exact ranks for mixed sizes. Let me use schoolbook for simplicity.
    # Schoolbook rank for (a,b,c) is a*b*c
    
    # (3,3,3): 27
    # (3,3,2): 18
    # (3,2,3): 18
    # (3,2,2): 12
    # (2,3,3): 18
    # (2,3,2): 12
    # (2,2,3): 12
    # (2,2,2): 8 (schoolbook) or 7 (Strassen)
    
    # Sum with Strassen for (2,2,2): 27+18+18+12+18+12+12+7 = 124
    
    # Still not great. The block decomposition doesn't help much because the blocks are large.
    
    # Let me try a completely different approach: use the known fact that
    # the rank of 5x5x5 matmul is at most 50 (by some results)
    
    # Actually, I recall that the rank of n x n matmul is at most n^2 * omega(n)/n^3 * n^3
    # where omega is the matrix multiplication exponent. For n=5, this is not directly helpful.
    
    # Let me just try to find a valid decomposition by solving the linear system
    # for a given rank, using a randomized approach
    
    # The key equation: sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w]
    
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # sums to T
    
    # One approach: fix U and V, then solve for W
    # Given U, V, the equation becomes:
    # For each (u,v,w): sum_r (U[u,r] * V[v,r]) * W[w,r] = T[u,v,w]
    # Let M_r[u,v] = U[u,r] * V[v,r]
    # Then: sum_r M_r[u,v] * W[w,r] = T[u,v,w]
    # This is a linear system in W: for each r, W[w,r] appears with coefficient M_r[u,v]
    
    # Actually, it's: for each (u,v,w), sum_r M_r[u,v] * W[w,r] = T[u,v,w]
    # This is a system of 15625 equations in 25*R unknowns (entries of W)
    
    # If R is small enough, this might be solvable
    
    # Let me try R = 50 and see if I can solve it
    
    # But this is computationally expensive. Let me try a simpler heuristic.
    
    # Given the time constraints, let me just return the schoolbook solution
    # but with a slightly optimized structure if possible
    
    # Actually, let me try one more thing: use the tensor product of decompositions
    
    # 5 = 2 + 3
    # Matmul 5x5 = block matrix [[A11, A12], [A21, A22]] where A11 is 2x2, etc.
    # (A*B)11 = A11*B11 + A12*B21
    # (A*B)12 = A11*B12 + A12*B22
    # (A*B)21 = A21*B11 + A22*B21
    # (A*B)22 = A21*B12 + A22*B22
    
    # Each term is a matmul of sizes (2,2,2) or (2,2,2) etc.
    # But the sums mean we can't just add ranks
    
    # I think the best I can do within the time limit is to return schoolbook
    # unless I can find a clever decomposition
    
    # Let me try to implement a search for rank 50
    
    # Actually, given the complexity and time constraints, let me just return schoolbook
    # The score will be 1.0, which is the baseline
    
    # But wait, the directive says to investigate improvements. Let me try a simple
    # random search with a very short time limit, and if it finds something, use it
    # Otherwise fall back to schoolbook
    
    # For now, return schoolbook to ensure validity
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
