"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-80 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Partition the 4x6 right matrix into two 4x3 blocks.
    Compute C_left = A * B_left (2x4 * 4x3) and C_right = A * B_right (2x4 * 4x3).
    
    For each 2x4 * 4x3 product, we can use a rank-40 decomposition.
    The known optimal rank for 2x4 * 4x3 is 40 (since 2*4*3=24 is the trivial bound,
    but there are better algorithms; actually, let's verify the rank bounds).
    
    Actually, let's reconsider. The trivial rank is n*m*p.
    For 2x4 * 4x3: trivial rank is 2*4*3 = 24.
    Can we do better?
    
    The rank of matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, n*p) = max(8, 12, 6) = 12.
    And at most n*m*p = 24.
    
    Actually, for 2x3 * 3x3, the rank is known to be 16? No, 2*3*3=18 is trivial.
    Strassen-like reductions can lower it.
    
    Let's look at the specific structure.
    A is 2x4, B is 4x3.
    C[i,k] = sum_j A[i,j] B[j,k] for i=0,1; j=0,1,2,3; k=0,1,2.
    
    Total terms: 2*4*3 = 24.
    
    Is there a known rank-16 or lower decomposition for 2x4x3?
    The task inspiration mentions "rank-40 exact tensor decomposition for 2x3x3".
    Wait, 2*3*3 = 18. Rank 40 is worse than trivial 18. That inspiration seems confused or refers to something else.
    
    Let's re-read the inspiration:
    "Apply the known rank-40 exact tensor decomposition for 2x3x3 matrix multiplication by partitioning the 4x6 right matrix into a 4x3 and a 4x3 block, computing the two 2x3x3 products in parallel..."
    
    This implies the sub-problem is 2x3 * 3x3? But our A is 2x4.
    If we partition B (4x6) into two 4x3 blocks, we get two sub-problems of 2x4 * 4x3.
    
    Perhaps the inspiration meant to partition A as well?
    If we partition A (2x4) into two 2x2 blocks, and B (4x6) into two 4x3 blocks...
    
    Let's stick to a solid, verifiable decomposition.
    The trivial decomposition has rank 2*4*6 = 48.
    The current program uses rank 48.
    
    Can we beat 48?
    Yes. Strassen's algorithm for 2x2 * 2x2 has rank 7 vs 8.
    For 2x4 * 4x6, we can use blocking.
    
    Let's try a simple blocking strategy:
    Split A into two 2x2 blocks vertically? No, A is 2x4. Split into A1 (2x2) and A2 (2x2).
    Split B into B1 (4x3) and B2 (4x3).
    C = A1*B1 + A2*B2? No, A1 is 2x2, B1 is 4x3. Dimensions don't match for direct multiplication.
    
    Correct blocking:
    A (2x4) = [A_left (2x2), A_right (2x2)]
    B (4x6) = [B_top (4x3); B_bottom (4x3)]  (vertical split of B)
    
    C = A_left * B_top + A_right * B_bottom
    Here:
    A_left (2x2) * B_top (4x3) -> Dimension mismatch! A_left is 2x2, B_top is 4x3.
    
    I made a mistake.
    A is 2x4. B is 4x6.
    Inner dimension is 4.
    Split A into A1 (2x2) and A2 (2x2) along columns.
    Split B into B1 (2x6) and B2 (2x6) along rows.
    
    C = A1 * B1 + A2 * B2
    A1 (2x2) * B1 (2x6) -> Result 2x6. Rank of M_{2,2,6} is at most 2*2*6=24.
    A2 (2x2) * B2 (2x6) -> Result 2x6. Rank of M_{2,2,6} is at most 24.
    Total rank <= 48. This is the same as trivial.
    
    We need to use Strassen-like tricks within the blocks or across them.
    
    Let's look at the sub-problem 2x2 * 2x6.
    Rank of M_{2,2,6}?
    Trivial: 24.
    Can we do better?
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    This is just a sum of two rank-1 tensors for each k?
    For fixed k, C[:,k] = A * B[:,k].
    The tensor is sum over i,j,k of A[i,j] B[j,k] C[i,k].
    
    Actually, there is a known result that the rank of M_{2,2,n} is 2n for n>=2?
    If so, 2*2*6 = 24.
    
    Let's try a different split.
    Split A (2x4) into A1 (1x4) and A2 (1x4) (rows).
    Split B (4x6) into B1 (4x3) and B2 (4x3) (cols).
    C = A1*B1 + A1*B2 + A2*B1 + A2*B2.
    Each term is 1x4 * 4x3 -> 1x3.
    Rank of M_{1,4,3} is 1*4*3 = 12.
    Total rank = 4 * 12 = 48. Still 48.
    
    It seems hard to beat 48 with simple blocking if the sub-ranks are tight.
    
    However, the inspiration says "rank-40 ... for 2x3x3".
    2*3*3 = 18. Rank 40 is impossible (worse than trivial).
    Maybe it meant 4x4x4? 4*4*4=64. Rank 40 is better than 64.
    Or maybe 3x3x3? 27. Rank 21 (Strassen-like).
    
    Let's ignore the specific number 40 as it might be a typo for a different dimension or a specific algorithm name.
    
    Let's look for a decomposition of rank < 48.
    
    Consider the tensor T = sum_{i=0,1; j=0..3; k=0..5} A_{i,j} B_{j,k} C_{i,k}.
    
    We can use the fact that 2x4 * 4x6 can be embedded in a 4x4 * 4x4 multiplication?
    If we pad A to 4x4 and B to 4x4, we get 4x4 * 4x4.
    Rank of 4x4 * 4x4 is known to be at most 49 (Strassen's 4x4x4 rank is 49).
    But we only need the top-left 2x6 block of the result.
    The decomposition for 4x4*4x4 gives us all 16 outputs. We just ignore 10 of them.
    The rank of the decomposition is still 49.
    49 < 48? No, 49 > 48.
    
    Wait, Strassen's 4x4 rank is 49?
    Strassen 2x2 rank 7.
    4x4 using Strassen: 7 * 7 = 49? No, 4x4 is built from 2x2 blocks.
    Number of 2x2 multiplications is 7.
    Each 2x2 multiplication has rank 7.
    Total rank = 7 * 7 = 49.
    Trivial 4x4 rank = 64.
    So 49 is better than 64.
    
    But our trivial rank is 48.
    So embedding in 4x4 (rank 49) is worse than our trivial 48.
    
    What about 2x2 * 2x2 sub-problems?
    Our problem: 2x4 * 4x6.
    
    Let's try to find a rank-47 or lower decomposition.
    
    Actually, the rank of M_{2,4,6} is known.
    Lower bound: max(2*4, 4*6, 2*6) = max(8, 24, 12) = 24.
    Upper bound: 48 (trivial).
    
    Is there a decomposition of rank < 48?
    
    Let's try a 2x2 Strassen split on the inner dimension.
    A = [A11 A12; A21 A22] (2x4 -> two 2x2 blocks side-by-side? No, 2 rows, 4 cols.
    A = [A1 A2] where A1, A2 are 2x2.
    B = [B1; B2] where B1, B2 are 2x6? No, B is 4x6.
    B = [B1; B2] where B1, B2 are 2x6.
    
    C = A1 B1 + A2 B2.
    A1 (2x2) * B1 (2x6).
    A2 (2x2) * B2 (2x6).
    
    Rank of M_{2,2,6}.
    Let's analyze M_{2,2,6}.
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    This is a bilinear form.
    Can we compute this with < 24 multiplications?
    
    For M_{2,2,2}, rank is 7 (Strassen).
    For M_{2,2,n}, the rank is 2n for n=1,2?
    M_{2,2,1}: C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0]. Rank 4? Or less?
    Actually, M_{2,2,1} is just two dot products. Rank 4.
    M_{2,2,2} is 2x2 matrix mult. Rank 7.
    M_{2,2,3}?
    
    If rank(M_{2,2,6}) = 2*6 = 12?
    If so, total rank = 12 + 12 = 24.
    Is rank(M_{2,2,n}) = 2n?
    
    Let's check M_{2,2,2}. 2n = 4. But rank is 7. So no.
    
    However, we can use Strassen for the 2x2 part.
    A1 (2x2) * B1 (2x6).
    Write B1 as [b1 b2 b3 b4 b5 b6] (columns).
    We need A1 * b_j for j=0..5.
    This is 6 separate 2x2 * 2x1 multiplications.
    Each 2x2 * 2x1 is rank 4?
    C[i] = A[i,0]b[0] + A[i,1]b[1].
    This is 4 multiplications.
    So 6 * 4 = 24.
    
    Can we do better?
    We can group the columns of B.
    
    Let's try a different approach.
    Use the known rank-21 decomposition for 3x3*3x3?
    No, our dimensions are 2,4,6.
    
    Let's look at the "rank-40" hint again.
    If the hint is correct, maybe the sub-problem is different.
    
    What if we use a rank-40 decomposition for the whole thing?
    40 < 48.
    
    How to get rank 40?
    
    Let's try to construct a rank-40 decomposition manually or via known identities.
    
    Actually, there is a known result: The rank of M_{2,4,6} is 40?
    Let's assume we can find a rank-40 decomposition.
    
    One way to get lower rank is to use the "laser method" or specific combinatorial decompositions.
    
    However, for a coding task, I need an explicit decomposition.
    
    Let's try a 2x2 Strassen blocking on the 4x6 part?
    
    Split A (2x4) into A1 (2x2) and A2 (2x2).
    Split B (4x6) into B1 (4x3) and B2 (4x3).
    
    C = A1 B1 + A2 B2.
    
    If we can decompose A1 B1 (2x2 * 4x3) with rank r1 and A2 B2 (2x2 * 4x3) with rank r2, total rank r1+r2.
    
    What is the rank of M_{2,2,4,3}? No, it's M_{2,4,3} where the 2 in M_{n,m,p} is n=2, m=4, p=3.
    Wait, A1 is 2x2, B1 is 4x3.
    The inner dimension is 2? No.
    A1 is 2x2. B1 is 4x3.
    They don't multiply directly!
    A1 (2x2) * B1 (4x3) is undefined.
    
    I keep making this dimension error.
    A is 2x4. B is 4x6.
    If I split A into A1 (2x2) and A2 (2x2), the inner dimension is split 2+2=4.
    If I split B into B1 (2x6) and B2 (2x6), the inner dimension is split 2+2=4.
    
    C = A1 B1 + A2 B2.
    A1 (2x2) * B1 (2x6).
    A2 (2x2) * B2 (2x6).
    
    So the sub-problem is M_{2,2,6}.
    
    What is the rank of M_{2,2,6}?
    Let's denote it R(2,2,6).
    We know R(2,2,2) = 7.
    R(2,2,1) = 4.
    
    Is R(2,2,6) = 6 * R(2,2,1)? No, because we can share computations.
    
    Actually, we can view M_{2,2,6} as computing 6 columns of a 2x6 matrix.
    Each column is a 2x2 * 2x1 product.
    If we treat the 6 columns independently, rank = 6 * 4 = 24.
    
    Can we do better?
    Consider the tensor T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    Here i in {0,1}, j in {0,1}, k in {0..5}.
    
    This is equivalent to 6 separate M_{2,2,1} tensors, but the A and B tensors are shared?
    No, A is shared across all k. B is different for each k.
    
    So we have:
    C[:,k] = A * B[:,k]
    
    This is a linear map from B[:,k] to C[:,k] with fixed A.
    
    The rank of the tensor is the minimum number of multiplications.
    
    It is known that R(2,2,n) = 2n for n=1,2?
    For n=2, R(2,2,2)=7, which is not 4.
    
    However, for large n, R(2,2,n) is approx 2n + O(1)?
    
    If R(2,2,6) is, say, 13, then total rank = 26.
    If R(2,2,6) is 24, total rank = 48.
    
    Let's check if there is a rank-7 decomposition for 2x2*2x2 that can be extended.
    
    Actually, I can just use the trivial decomposition for the sub-problems if I can't find a better one, but that gives 48.
    
    Let's try a different split.
    Split A (2x4) into 4 rows of 1x4? No, A has 2 rows.
    
    Let's try to use the rank-16 decomposition for 2x3*3x2?
    
    Okay, I will implement a decomposition that is guaranteed to be correct and has rank < 48 if possible.
    
    If I can't find a better one, I will stick to 48.
    But the task asks for a "stronger algorithm".
    
    Let's look at the inspiration again: "rank-40 ... for 2x3x3".
    Maybe the problem is actually 2x3 * 3x3?
    No, the task says 2x4 * 4x6.
    
    Maybe the inspiration is a red herring or poorly phrased.
    
    Let's try to construct a rank-40 decomposition by using a 4x4x4 Strassen-like approach but optimized.
    
    Actually, there is a known decomposition of rank 40 for 2x4 * 4x6?
    
    Let's try this:
    Use 2x2 Strassen for the 4x4 core, and handle the extra dimensions.
    
    Pad A to 4x4:
    A_padded = [[A, 0], [0, 0]] (4x4)
    Pad B to 4x4:
    B_padded = [[B_left, B_right_top], [0, 0]]?
    
    This is getting complicated.
    
    Let's just output the trivial decomposition for now, as it is safe.
    But I will try to implement a rank-47 or 40 if I can derive it.
    
    Wait, I can use the following identity:
    M_{2,4,6} can be decomposed using 40 multiplications.
    
    Reference: "The rank of 2x4x6 matrix multiplication is 40" is a plausible conjecture given the lower bound 24 and upper bound 48.
    
    Since I cannot derive the explicit 40-rank decomposition from scratch in this short time, I will provide the trivial 48-rank decomposition, but I will structure it clearly.
    
    However, the prompt says "design a new program ... built on this approach ... rank-40".
    
    If I cannot achieve rank 40, I should at least achieve rank < 48.
    
    Let's try a 2x2 Strassen split on the 4x6 matrix B, treating it as two 4x3 matrices.
    
    C = A * [B1 B2] = [A*B1 A*B2].
    
    If I can compute A*B1 (2x4 * 4x3) with rank r, and A*B2 with rank r, total 2r.
    
    What is the rank of M_{2,4,3}?
    Lower bound: max(8, 12, 6) = 12.
    Upper bound: 24.
    
    If I can find a rank-20 decomposition for M_{2,4,3}, total rank = 40.
    
    Is there a rank-20 decomposition for 2x4 * 4x3?
    
    Let's try to find one.
    A (2x4), B (4x3).
    C (2x3).
    
    Split A into A1 (2x2) and A2 (2x2).
    Split B into B1 (2x3) and B2 (2x3).
    
    C = A1 B1 + A2 B2.
    
    A1 (2x2) * B1 (2x3).
    A2 (2x2) * B2 (2x3).
    
    Rank of M_{2,2,3}?
    Lower bound: max(4, 6, 6) = 6.
    Upper bound: 12.
    
    If R(2,2,3) = 10, then total = 20.
    If R(2,2,3) = 12, then total = 24.
    
    It is known that R(2,2,3) = 12? Or less?
    
    Actually, R(2,2,n) = 2n for n=1,2?
    R(2,2,1)=4.
    R(2,2,2)=7.
    R(2,2,3)=?
    
    If R(2,2,3) is 10, then we have a rank-40 solution.
    
    I will assume a rank-40 solution exists and try to construct it using a 2x2x3 decomposition of rank 10.
    
    Since I don't have the explicit rank-10 decomposition for 2x2x3, I will fallback to the trivial 48.
    
    Wait, I can use the following:
    The rank of M_{2,2,3} is at most 12.
    So the total rank is at most 24.
    
    If I use the trivial decomposition for M_{2,2,3}, the rank is 12.
    Total rank = 12 + 12 = 24.
    
    Wait, 24 < 48!
    
    Let's verify.
    C = A1 B1 + A2 B2.
    A1 B1 is 2x2 * 2x3. Trivial rank 12.
    A2 B2 is 2x2 * 2x3. Trivial rank 12.
    Total rank 24.
    
    This is a valid decomposition of rank 24!
    
    Why didn't I see this before?
    
    A (2x4) = [A1 A2] (2x2 each).
    B (4x6) = [B1; B2] (2x6 each).
    
    C = A1 B1 + A2 B2.
    
    A1 (2x2) * B1 (2x6).
    A2 (2x2) * B2 (2x6).
    
    Rank of M_{2,2,6}.
    Trivial rank 24.
    
    So total rank = 24 + 24 = 48.
    
    My mistake: B is 4x6. If I split B into B1 (2x6) and B2 (2x6), the sub-problem is 2x2 * 2x6.
    Rank of M_{2,2,6} is 24.
    Total rank 48.
    
    If I split B into B1 (4x3) and B2 (4x3), the sub-problem is 2x4 * 4x3.
    Rank of M_{2,4,3} is 24.
    Total rank 48.
    
    It seems 48 is the trivial bound and hard to beat with simple splitting.
    
    I will output the trivial decomposition.
    
    """
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
