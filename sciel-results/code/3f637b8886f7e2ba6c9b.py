"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen-style block decomposition for 4x4x4, then extend to 4x4x6.
    # For 4x4x4, rank 18 is known (Strassen for 2x2x2 composed).
    # For 4x4x6, we can use a block approach.
    
    # Strategy: Decompose the 4x4x6 multiplication using known low-rank decompositions.
    # The 4x4x4 matrix multiplication has rank 18 (via Strassen composition).
    # For 4x4x6, we can think of it as two 4x4x3 blocks or use a different approach.
    
    # Let's try a direct approach: use the known 4x4x4 rank-18 decomposition 
    # and extend it to handle the extra 2 columns in the p dimension.
    
    # Actually, let's use a simpler but effective approach:
    # The 4x4x4 multiplication can be done with rank 18.
    # For 4x4x6, we need to multiply A(4x4) by B(4x6) to get C(4x6).
    # We can decompose B into two 4x3 matrices and use appropriate decompositions.
    
    # Let's use a block Strassen approach:
    # Split A into 2x2 blocks of 2x2 matrices, B into 2x3 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    
    # C = A*B, C is 4x6, split into 2x3 blocks of 2x2:
    # C = [[C11, C12, C13], [C21, C22, C23]]
    
    # Using Strassen for 2x2x2 (rank 7) for each block combination...
    # But we need to be careful about the tensor structure.
    
    # Alternative: Use the fact that 4x4x4 has rank 18, and for 4x4x6 we can
    # potentially use rank 18 + something for the extra columns.
    
    # Let me try a different approach: use a known rank decomposition.
    # For m x n x p matrix multiplication, the schoolbook rank is m*n*p.
    # For 4x4x6, schoolbook is 96.
    
    # Let's use a block decomposition with Strassen:
    # The 4x4x4 multiplication uses 18 multiplications (Strassen composition).
    # For 4x4x6, we can view it as computing 4x4 times 4x6.
    
    # Actually, let's just use a well-known approach: 
    # Decompose using 2x2x2 Strassen blocks.
    
    # A is 4x4, B is 4x6, C is 4x6.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13], [B21, B22, B23]] (each 2x2)
    # C = [[C11, C12, C13], [C21, C22, C23]] (each 2x2)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # Each product of two 2x2 matrices can be done with 7 scalar multiplications (Strassen).
    # But we need to share computations where possible.
    
    # Let's define the Strassen products for each pair:
    # For A11*B11: M1 = (A11+A22)*(B11+B22), M2 = (A21+A22)*B11, 
    # M3 = A11*(B12-B22), M4 = A22*(B21-B11), M5 = (A11+A12)*B22,
    # M6 = (A21-A11)*(B11+B12), M7 = (A12-A22)*(B21+B22)
    # Then: C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # This gets complex. Let me use a simpler but still good decomposition.
    
    # For 4x4x4, rank 18 is achievable. For 4x4x6, let's estimate:
    # We can use 3 copies of 4x4x2 (each might have lower rank) or 
    # use a direct tensor decomposition.
    
    # Let's use the following approach: 
    # The multiplication tensor for 4x4x6 has a natural structure.
    # We'll use a block approach where we split the p-dimension.
    
    # For now, let's use a rank that's better than schoolbook but achievable.
    # 4x4x4 has rank 18. For 4x4x6, a reasonable upper bound is 18*3/2 = 27 
    # if we can share some computations, or simply 18+18 = 36 if we treat it as two 4x4x3.
    
    # Actually, let's use a known result: the rank of 4x4x4 is 18.
    # For 4x4x6, we can potentially use rank around 24-30.
    
    # Let me implement a specific decomposition:
    # Use Strassen for the 2x2x2 blocks with sharing.
    
    # I'll use a straightforward but optimized block decomposition.
    
    # For each of the 6 output columns (in 2x2 blocks), we need to compute
    # the product of 2x2 blocks. Let's use 7 multiplications per 2x2x2 product.
    
    # There are 2 (rows of A blocks) * 3 (cols of B blocks) = 6 pairs of 2x2x2 multiplications.
    # But we can share some Strassen intermediates.
    
    # Let's just use a rank of 42 (6 * 7) as a safe upper bound with Strassen per block.
    # This is better than schoolbook 96.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Indexing:
    # U[i*m+j, c] for i in 0..3, j in 0..3
    # V[j*p+k, c] for j in 0..3, k in 0..5
    # W[k*n+i, c] for k in 0..5, i in 0..3
    
    # Strassen for 2x2x2:
    # Given A (2x2), B (2x2), C = A*B
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # 
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Now for our block structure:
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
    #            A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B13 (rows 0-1, cols 4-5)
    #            B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3), B23 (rows 2-3, cols 4-5)
    
    # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3), C13 (rows 0-1, cols 4-5)
    #            C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3), C23 (rows 2-3, cols 4-5)
    
    # For each pair (Ai, Bj), we compute their 2x2x2 product using 7 multiplications.
    # Total: 2*3 = 6 pairs, each with 7 mults = 42 mults.
    
    # Let's define the 6 pairs:
    # Pair 0: A11 * B11 -> contributes to C11
    # Pair 1: A11 * B12 -> contributes to C12
    # Pair 2: A11 * B13 -> contributes to C13
    # Pair 3: A21 * B11 -> contributes to C21
    # Pair 4: A21 * B12 -> contributes to C22
    # Pair 5: A21 * B13 -> contributes to C23
    # Pair 6: A12 * B21 -> contributes to C11
    # Pair 7: A12 * B22 -> contributes to C12
    # Pair 8: A12 * B23 -> contributes to C13
    # Pair 9: A22 * B21 -> contributes to C21
    # Pair 10: A22 * B22 -> contributes to C22
    # Pair 11: A22 * B23 -> contributes to C23
    
    # Wait, that's 12 pairs, not 6. Let me reconsider.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # So we have 12 individual 2x2x2 multiplications, each using 7 scalar mults = 84 total.
    # That's not better than schoolbook 96 by much.
    
    # Let me use a different approach. For 4x4x4, the rank is 18.
    # Let me just use a known rank-18 decomposition for 4x4x4 and extend it.
    
    # Actually, for simplicity and correctness, let me use a rank that's definitely achievable.
    # The 4x4x4 multiplication has rank 18. For 4x4x6, we can use a block approach.
    
    # Let me just use schoolbook for now but with a smaller constant by using 
    # a smarter decomposition. Or better, let me use a known low-rank decomposition.
    
    # For 4x4x4, one known rank-18 decomposition exists. Let me implement it.
    # But for 4x4x6, I need to adapt it.
    
    # Let's use a simpler approach: rank 24 for 4x4x6 using a block method.
    
    # Actually, let me just use a valid decomposition with rank 48 (2x schoolbook per dim).
    # No, let me use the Strassen approach properly.
    
    # For 4x4x4 with Strassen: split into 2x2 blocks, use 7 mults per 2x2x2.
    # 4 blocks of 2x2x2, but with sharing: actually 4*7 = 28, but with sharing it's 18.
    
    # For 4x4x6, let me use rank 42 (6*7) as a safe bound.
    
    # Let me implement the 12-pair Strassen approach with rank 84, which is valid.
    # Actually that's worse than I thought. Let me just use a clean decomposition.
    
    # Let me use rank 48: for each of the 4 output rows, compute using a 4x4x6 approach.
    
    # I'll just use a straightforward valid decomposition with a reasonable rank.
    # Let's use rank 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use a block approach: for each output position (i,k), 
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We can use a decomposition that groups these.
    
    # Let me use a simple but effective decomposition:
    # For each i (0..3) and k (0..5), we need sum over j (0..3).
    # Total terms: 4*6*4 = 96 (schoolbook).
    
    # Let me use a rank-48 decomposition by pairing up some terms.
    
    # Actually, let me just use the schoolbook decomposition but that's rank 96.
    # Let me try to find a better one.
    
    # For 4x4x4, rank 18 is known. Let me use a known decomposition.
    # The 4x4x4 multiplication tensor can be decomposed with 18 terms.
    
    # Let me use a different strategy: use the fact that we can decompose 4x4x6
    # as a combination of smaller decompositions.
    
    # I'll use a rank of 36 as a reasonable target (3x the 4x4x4 rank of 12? No, 18).
    
    # Let me just implement a valid decomposition. I'll use rank 48 with a block method.
    
    # For each pair (i, k) where i is row of A and k is col of B:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # I can use a decomposition where I group j values.
    
    # Let me use a simple valid decomposition with rank 48:
    # For each i (4 values) and each group of 3 k values (2 groups),
    # use a 4x4x3 decomposition.
    
    # Actually, let me just use the schoolbook but that's rank 96.
    
    # Let me use a rank-48 decomposition based on the following:
    # For each i (0..3), we compute row i of C.
    # Row i of C is A[i,:] * B, which is a 1x4 times 4x6 = 1x6.
    # This can be done with rank 4 (one per j) for each i, total 16.
    # But that's not how tensor decomposition works directly.
    
    # Let me just use a valid decomposition. I'll use rank 48.
    
    # For simplicity and correctness, let me use the schoolbook decomposition
    # but with a reduced rank by exploiting structure.
    
    # Actually, the simplest correct approach: use rank = n*m*p = 96 (schoolbook).
    # But let me try to do better.
    
    # Let me use a rank-36 decomposition if possible, otherwise fall back to 96.
    
    # For now, let me use a valid decomposition with rank 48.
    
    # I'll construct it as follows:
    # Use 4x4x4 rank-18 decomposition for the first 4 columns, and 
    # 4x4x2 decomposition for the last 2 columns.
    
    # 4x4x2: rank is at most 8 (schoolbook). So total rank = 18 + 8 = 26.
    
    # Let me implement a 4x4x4 rank-18 decomposition and a 4x4x2 rank-8 decomposition.
    
    # For 4x4x2, schoolbook is 4*4*2 = 32, but we can do better.
    # Actually, 4x4x2 rank is at most 8? No, that's not right.
    # The rank of m x n x p multiplication is at most m*n*p, and for 4x4x2 it's at most 32.
    # But we can use a block decomposition.
    
    # Let me just use a safe decomposition. I'll use rank 48.
    
    # Here's a simple valid decomposition with rank 48:
    # For each i (0..3) and each k (0..5), use 2 terms instead of 4.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # We can write this as:
    # = (A[i,0]+A[i,1])*B[0,k] + (A[i,2]+A[i,3])*B[1,k] + A[i,0]*B[0,k] + ...
    # This doesn't obviously reduce the rank.
    
    # Let me just use the schoolbook decomposition with rank 96.
    # It's correct and valid.
    
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
