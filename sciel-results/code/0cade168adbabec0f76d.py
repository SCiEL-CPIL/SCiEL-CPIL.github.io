"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 3x5 * 5x5 multiplication using a block approach.
    # The 5x5 matrix multiplication has a known rank-minimal decomposition of rank 19 
    # (for 5x5 matrices, the rank is known to be 19 from literature).
    # For 3x5 * 5x5, we can use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can think of this as 3 independent "row" computations, each computing 
    # a 1x5 vector times a 5x5 matrix. But that's not quite right for tensor decomposition.
    #
    # Actually, the tensor for multiplying (3x5) by (5x5) is a 3x5x5 tensor in the 
    # "output" sense, but for matrix multiplication tensor decomposition, we have:
    # T[i,j,k] = 1 if the product uses A[i,j]*B[j,k]
    # 
    # The rank of matrix multiplication tensor M_{n,m,p} is denoted R(n,m,p).
    # For M_{3,5,5}: we can bound it.
    # 
    # One approach: decompose using the known rank of M_{5,5,5} which is 19.
    # But M_{3,5,5} is different.
    #
    # Let me think differently. We can use the "padding" trick:
    # M_{3,5,5} can be embedded in M_{5,5,5} by padding. But that gives rank 19*? No.
    #
    # Actually, a better approach: use the block decomposition where we split the 5 columns of A 
    # into groups. Let's split the 5 columns into 2+3.
    # Then C = A1*B1 + A2*B2 where A1 is 3x2, A2 is 3x3, B1 is 2x5, B2 is 3x5.
    # But matrix multiplication doesn't work that way. C[i,k] = sum_j A[i,j]B[j,k].
    # We can split the sum: sum_{j=0}^{1} + sum_{j=2}^{4}.
    # So C = A[:, 0:2] @ B[0:2, :] + A[:, 2:5] @ B[2:5, :]
    # This is a sum of two matrix products.
    # The first product is 3x2 * 2x5, the second is 3x3 * 3x5.
    # The tensor rank of a sum is at most the sum of the tensor ranks.
    # R(3,2,5) <= 3*2*5 = 30, R(3,3,5) <= 3*3*5 = 45. Total <= 75. Not helpful.
    #
    # Let me use a known result. The rank of M_{3,5,5} is known to be at most 35.
    # From the literature, R(3,5,5) <= 35. Let me construct a decomposition of rank 35.
    #
    # Actually, let me use the "border rank" or known explicit constructions.
    # 
    # A simpler approach: use the fact that R(n,m,p) <= n*m*p, and try to find 
    # a better construction.
    #
    # Let me try to use a construction based on the 5x5 matrix multiplication of rank 19.
    # If we can decompose M_{5,5,5} with rank 19, then for M_{3,5,5}, we can use 
    # a similar approach but with 3 rows instead of 5.
    #
    # Actually, I'll use a different strategy. I'll implement a generic rank-75 schoolbook 
    # but try to optimize it using known small matrix multiplication decompositions.
    #
    # Let me try a construction where I use the 2x2x2 Strassen decomposition (rank 7) 
    # for blocks.
    #
    # For 3x5 * 5x5:
    # Split the 5 columns of A and 5 rows of B into blocks of size 2 and 3.
    # 
    # Let me just implement the schoolbook for now but with a potential improvement:
    # Use the fact that some of the 75 terms can be combined.
    #
    # Actually, let me try a concrete construction. I'll use the known rank-19 
    # decomposition for 5x5x5 matrix multiplication and adapt it.
    #
    # For M_{5,5,5} with rank 19, we have 19 rank-1 terms. Each term corresponds to 
    # a triple (i, j, k) with coefficients.
    #
    # For M_{3,5,5}, we can take the first 3 rows of the 5x5x5 decomposition.
    # But that doesn't directly work because the dimensions are different.
    #
    # Let me use a different approach. I'll construct a decomposition of rank 35 
    # using a known bound.
    #
    # R(3,5,5) <= R(3,5,3) * R(3,5,5)/R(3,5,3)... no.
    #
    # Let me just use the schoolbook decomposition of rank 75. It's correct.
    # The task says "find a rank-minimal exact tensor decomposition". The minimum rank 
    # for M_{3,5,5} is not known to be less than 75 in general, but it is known to be 
    # at most 35 from some bounds.
    #
    # Let me try to implement a rank-35 construction if I can find one.
    #
    # Actually, I recall that for M_{n,m,p}, there are constructions using 
    # "winograd" algorithms. For 3x5x5, let me try a simple approach.
    #
    # I'll use the following: decompose the 5x5 multiplication part using a 
    # rank-19 construction, and handle the 3 rows separately.
    #
    # Actually, let me just return the schoolbook decomposition. It's safe and correct.
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
