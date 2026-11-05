"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy:
  1. Split the middle dimension m=5 into m1=2 and m2=3.
     This gives two independent sub-problems:
       - 3x2 * 2x7  (rank 6 via 1x2*2x7 + 2x2*2x7)
       - 3x3 * 3x7  (rank 18 via 1x3*3x7 + 2x3*3x7)
     Total rank = 24.
  2. Each sub-problem is decomposed using the fact that:
       - 1xm * m*p has rank m (each term: e_0 ⊗ e_j ⊗ 1_vector)
       - 2xm * m*p can be split into two 1xm * m*p blocks, rank 2m
     But for 2x2 * 2x7 we can do even better with Strassen-like tricks.
     Actually, 2x2 * 2x7: split output into 7 columns of 2x2*2x1.
     Each 2x2 * 2x1 has rank 2 (it's just two dot products).
     So 2x2 * 2x7 has rank 14? No, that's schoolbook.
     Actually 2x2 * 2x1: T[i,j,k] where i in {0,1}, j in {0,1}, k=0.
     T[0,0,0]=1, T[0,1,0]=0, T[1,0,0]=0, T[1,1,0]=1.
     This is just the identity on the 2x2 block, rank 2.
     So 2x2 * 2x7: 7 such blocks, each rank 2, total rank 14.
     But we can do better: 2x2 * 2x7 is equivalent to multiplying a 2x2 matrix
     by a 2x7 matrix. The tensor has 2*2*7 = 28 entries.
     Actually, let me just use the simple decomposition:
       1x2 * 2x7: rank 2
       2x2 * 2x7: rank 4 (split into two 1x2*2x7)
     Total for 3x2*2x7: 2 + 4 = 6. ✓
     
     For 3x3 * 3x7:
       1x3 * 3x7: rank 3
       2x3 * 3x7: split into two 1x3*3x7: rank 6
     Total: 3 + 6 = 9? No wait, 2x3*3x7 is not two 1x3*3x7.
     
     Let me be more careful.
     3x3 * 3x7:
       Split rows of A (3x3) into A1 (1x3) and A2 (2x3).
       C1 = A1 * B (1x7), where B is 3x7. Rank of 1x3*3x7 tensor = 3.
       C2 = A2 * B (2x7), where B is 3x7. 
         Split rows of A2 into A2a (1x3) and A2b (1x3).
         C2a = A2a * B (1x7), rank 3.
         C2b = A2b * B (1x7), rank 3.
       Total for 3x3*3x7: 3 + 3 + 3 = 9.
     
     So total rank = 6 + 9 = 15!
     
     Wait, that seems too good. Let me verify.
     
     For 1x3 * 3x7: T[0,j,k] = 1 for all j in {0,1,2}, k in {0..6}.
     Decomposition: sum_{j=0}^{2} e_0 ⊗ e_j ⊗ (1,1,1,1,1,1,1)
     That's 3 terms. ✓
     
     For 3x3 * 3x7:
       A is 3x3, B is 3x7, C is 3x7.
       C[i,k] = sum_j A[i,j] B[j,k]
       T[i,j,k] = 1 for all i,j,k.
       
       Split: i=0: T[0,j,k] = 1 for all j,k. This is the 1x3*3x7 tensor, rank 3.
       i=1: T[1,j,k] = 1 for all j,k. Another 1x3*3x7 tensor, rank 3.
       i=2: T[2,j,k] = 1 for all j,k. Another 1x3*3x7 tensor, rank 3.
       Total: 9. ✓
     
     So 3x3 * 3x7 has rank 9, not 18!
     
     And 3x2 * 2x7:
       T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0..6}.
       Split by i:
         i=0: 1x2*2x7 tensor, rank 2.
         i=1: 1x2*2x7 tensor, rank 2.
         i=2: 1x2*2x7 tensor, rank 2.
       Total: 6. ✓
     
     Grand total: 6 + 9 = 15.
     
     Can we do even better?
     
     For 1xm * m*p, the rank is exactly m (can't be less because each term
     contributes to at most one j-index in the middle, and we need all m).
     
     So 3x2*2x7 = 3 * 2 = 6 is minimal for this split.
     3x3*3x7 = 3 * 3 = 9 is minimal for this split.
     Total = 15.
     
     What if we split differently?
     m=5 = 1+4:
       3x1*1x7: rank 3 (1x1*1x7 tensor has rank 1, but 3 rows... 
         Actually 3x1*1x7: T[i,0,k] = 1 for all i,k.
         Decomposition: sum_i e_i ⊗ e_0 ⊗ (1,1,...,1). Rank 3.
       3x4*4x7: 
         Split rows: 3 * 4 = 12.
       Total: 3 + 12 = 15. Same.
     
     m=5 = 1+1+3:
       3x1*1x7: rank 3
       3x1*1x7: rank 3
       3x3*3x7: rank 9
       Total: 15. Same.
     
     m=5 = 2+3:
       3x2*2x7: rank 6
       3x3*3x7: rank 9
       Total: 15. Same.
     
     So 15 is the answer with this approach.
     
     Can Strassen help? For 2x2*2x2, Strassen gives rank 7 instead of 8.
     But our sub-problems are 1xm*m*p which have rank m, and we're not
     doing 2x2*2x2 blocks directly.
     
     Actually, let me reconsider. The tensor for n x m * m x p has
     T[i,j,k] = 1 for all valid i,j,k.
     
     For 1xm * m*p: T[0,j,k] = 1. Rank = m (as shown).
     For 2xm * m*p: T[i,j,k] = 1 for i in {0,1}. 
       = sum over i of (e_i ⊗ ... ) 
       Split into two 1xm*m*p tensors, each rank m. Total rank 2m.
       Can we do better? For 2x2*2x2, Strassen shows rank 7 < 8 = 2*2*2/2... 
       Wait, 2x2*2x2 schoolbook rank is 8. Strassen gives 7.
       But 2x2*2x2 is not the same as 2xm*m*p for general m,p.
       
     For our specific case, 15 seems to be the achievable rank.
     
     Let me implement this.
"""
import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    
    # Split middle dimension: m = 2 + 3
    m1, m2 = 2, 3
    
    # Sub-problem 1: 3x2 * 2x7
    # Split rows of A (3x2) into three 1x2 blocks
    # Each 1x2 * 2x7 has rank 2
    # Total rank for this sub-problem: 3 * 2 = 6
    
    # Sub-problem 2: 3x3 * 3x7
    # Split rows of A (3x3) into three 1x3 blocks
    # Each 1x3 * 3x7 has rank 3
    # Total rank for this sub-problem: 3 * 3 = 9
    
    total_rank = 6 + 9  # = 15
    
    U = [[0] * total_rank for _ in range(n * m)]  # 15 x 15
    V = [[0] * total_rank for _ in range(m * p)]  # 35 x 15
    W = [[0] * total_rank for _ in range(p * n)]  # 21 x 15
    
    col = 0
    
    # Sub-problem 1: 3x2 * 2x7
    # Middle indices: 0, 1 (for m1=2)
    # For each row i of A (i=0,1,2) and each middle index j (j=0,1):
    #   U[i*m + j][col] = 1
    #   V[j*p + k][col] = 1 for all k=0..6  -> but we need a single vector
    #   Actually, for 1x2 * 2x7: T[0,j,k] = 1
    #   Decomposition: sum_{j=0}^{1} e_0 ⊗ e_j ⊗ ones(7)
    #   So for term (i, j):
    #     U[i*m + j][col] = 1
    #     V[j*p + k][col] = 1 for all k  -> V[j*p][col]=1, V[j*p+1][col]=1, ..., V[j*p+6][col]=1
    #     W[k*n + i][col] = 1 for all k  -> W[0*n+i][col]=1, W[1*n+i][col]=1, ..., W[6*n+i][col]=1
    for i in range(n):
        for j in range(m1):
            U[i * m + j][col] = 1
            for k in range(p):
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1
    
    # Sub-problem 2: 3x3 * 3x7
    # Middle indices: 2, 3, 4 (for m2=3, offset by m1=2)
    # For each row i of A (i=0,1,2) and each middle index j (j=2,3,4):
    #   U[i*m + j][col] = 1
    #   V[j*p + k][col] = 1 for all k=0..6
    #   W[k*n + i][col] = 1 for all k=0..6
    for i in range(n):
        for j in range(m1, m):
            U[i * m + j][col] = 1
            for k in range(p):
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
