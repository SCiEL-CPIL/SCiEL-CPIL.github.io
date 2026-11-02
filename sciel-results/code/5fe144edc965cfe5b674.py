"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7
    
    # Schoolbook as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        # Check tensor identity: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta
        # Actually, the tensor T_{n,m,p} has entries T[i,j,k] = 1 for all i,j,k
        # The decomposition says T = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # So T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need this to equal 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != 1:
                        return False
        return True
    
    # Strategy: Use Strassen-like decomposition on 2x2 sub-blocks
    # For M_{3,5,7}, split as follows:
    # A (3x5) = [A_top (2x5); A_bottom (1x5)]
    # B (5x7) = [B_left (5x5) | B_right (5x2)]
    # C (3x7) = [A_top * B_left  A_top * B_right; A_bottom * B_left  A_bottom * B_right]
    #
    # For the 2x5 * 5x5 block: split B_left into [B_left1 (5x2) | B_left2 (5x2) | B_left3 (5x1)]
    # This gives M_{2,5,2} + M_{2,5,2} + M_{2,5,1}
    # M_{2,5,2} can potentially use Strassen-like methods
    #
    # Actually, let's try a direct approach: embed into a larger problem.
    # 
    # Alternative: Use the fact that M_{2,2,2} has rank 7 (Strassen).
    # We can decompose M_{3,5,7} by grouping indices.
    #
    # Let's try: split n=3 as 2+1, m=5 as 2+2+1, p=7 as 2+2+2+1
    # This gives many sub-problems, but most are 1-dimensional in some factor.
    #
    # A cleaner approach: use block Strassen.
    # View A as 2x5 stacked with 1x5, B as 5x5 next to 5x2.
    # The 2x5 * 5x5 part: we can split the 5 columns of B into 2+2+1.
    # For each 2x5 * 5x2 sub-product, we can use a Strassen-like method.
    #
    # M_{2,m,2}: The tensor rank. For m=2, it's 7 (Strassen).
    # For general m, M_{2,m,2} can be seen as: 
    # C[i,k] = sum_j A[i,j] B[j,k] for i,k in {0,1}, j in {0,...,m-1}
    # This is equivalent to computing a 2x2 product from m-dimensional vectors.
    # We can write A = sum_j a_j e_j^T, B = sum_j f_j b_j^T where a_j, f_j are 2-vectors
    # C = sum_j (a_j b_j^T) ... no, that's not right.
    # C = A B = (sum_j a_j e_j^T)(sum_l f_l b_l^T) = sum_{j,l} a_j (e_j^T f_l) b_l^T
    # But e_j^T f_l is a scalar, so C = sum_j a_j (sum_l (e_j^T f_l) b_l^T)
    # This doesn't immediately give a low-rank decomposition.
    #
    # Actually, M_{2,m,2} has tensor rank at most 4m (schoolbook: 2*m*2 = 4m).
    # For m=2, Strassen gives 7 < 8.
    # For m > 2, the rank is between some bounds.
    #
    # Let's just try to use Strassen on the 2x2x2 blocks where possible.
    # Split m=5 into 2+2+1.
    # Split p=7 into 2+2+2+1.
    # Split n=3 into 2+1.
    #
    # The 2x2x2 blocks use Strassen (rank 7 instead of 8).
    # Number of 2x2x2 blocks: depends on how we group.
    #
    # If we split n=3 into 2+1, m=5 into 2+2+1, p=7 into 2+2+2+1:
    # Blocks: (2,2,2) x 3 = 3 blocks of size 2x2x2
    # Plus blocks involving the 1-dim parts.
    #
    # This is getting complex. Let's try a simpler strategy:
    # Just use the schoolbook decomposition, which is guaranteed correct.
    #
    # For now, return schoolbook. Future work: implement explicit Strassen-based
    # decomposition for specific sub-blocks.
    
    # Attempt a simple improvement: check if any known decomposition works
    # For M_{3,5,7}, the rank is known to be at most 105 (schoolbook).
    # Literature suggests it could be lower, but explicit constructions are hard.
    
    # Return schoolbook decomposition
    return schoolbook()
# EVOLVE-BLOCK-END
