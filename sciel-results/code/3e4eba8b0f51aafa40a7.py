"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook rank = 3*3*8 = 72
    # We'll try to construct a rank-68 decomposition using Strassen on a 2x2 block of A.
    
    # Block decomposition of A (3x3):
    # A11 = A[0:2, 0:2] (2x2)
    # A12 = A[0:2, 2:3] (2x1)
    # A21 = A[2:3, 0:2] (1x2)
    # A22 = A[2:3, 2:3] (1x1)
    #
    # B (3x8) partitioned as:
    # B1 = B[0:2, :] (2x8)
    # B2 = B[2:3, :] (1x8)
    #
    # C (3x8) = A*B partitioned as:
    # C1 = C[0:2, :] = A11*B1 + A12*B2  (2x8)
    # C2 = C[2:3, :] = A21*B1 + A22*B2  (1x8)
    
    # Decompositions:
    # 1. A11*B1: 2x2x8 multiplication. Use Strassen for 2x2 (rank 7 per column), so 7*8 = 56 rank.
    #    But wait: Strassen applies to the 2x2 matrix product. For each column of B1, 
    #    we compute 2x2 times 2x1 using 7 multiplications.
    #    Actually, for M_{2,2,8}: we can think of it as 8 independent M_{2,2,1} computations.
    #    M_{2,2,1} has rank 7 (Strassen). So total rank = 8 * 7 = 56.
    #
    # 2. A12*B2: 2x1x8 multiplication. This is just element-wise: for each k, 
    #    C1[:,k] += A12 * B2[k]. This is rank 2*1*8 = 16.
    #
    # 3. A21*B1: 1x2x8 multiplication. Rank 1*2*8 = 16.
    #
    # 4. A22*B2: 1x1x8 multiplication. Rank 1*1*8 = 8.
    #
    # Total rank = 56 + 16 + 16 + 8 = 96. That's worse than schoolbook!
    
    # Wait, I made an error. Let me reconsider.
    # For A11*B1 where A11 is 2x2 and B1 is 2x8:
    # The product is (A11 * B1) which is 2x8.
    # We can write this as: for each column k of B1 (which is a 2x1 vector),
    # compute A11 * b_k using Strassen (7 mults).
    # So yes, 8 columns * 7 = 56.
    
    # Alternative: Maybe I should use Strassen differently.
    # Actually, let me reconsider the block sizes.
    
    # What if I partition A as 1x1 + 2x2 blocks differently?
    
    # Let me try: A = [[A11, A12], [A21, A22]] where A11 is 1x1, A12 is 1x2, A21 is 2x1, A22 is 2x2.
    # Then:
    # C = A*B where B is 3x8, partitioned as B = [B1; B2] where B1 is 1x8, B2 is 2x8.
    # C1 = C[0, :] = A11*B1 + A12*B2  (1x8)
    # C2 = C[1:3, :] = A21*B1 + A22*B2  (2x8)
    
    # 1. A11*B1: 1x1x8, rank 8
    # 2. A12*B2: 1x2x8, rank 16
    # 3. A21*B1: 2x1x8, rank 16
    # 4. A22*B2: 2x2x8, rank 56 (using Strassen per column)
    # Total: 8+16+16+56 = 96. Still worse.
    
    # Hmm, Strassen doesn't help here because the 2x2 block needs to multiply 8 different vectors.
    
    # Let me think differently. The schoolbook rank is 72.
    # Can we do better than 72?
    
    # For M_{3,3,8}, the lower bound is at least max(3*3, 3*8, 3*8) = 24.
    # The trivial upper bound is 72.
    
    # Actually, let me just return the schoolbook decomposition for now, which is valid.
    
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
