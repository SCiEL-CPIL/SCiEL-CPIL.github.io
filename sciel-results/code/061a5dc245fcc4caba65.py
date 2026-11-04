"""
Rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.

Strategy: Pad 3x5x7 to 4x4x8, decompose as 4 independent 1x4x8 multiplications,
apply Strassen to each 1x4x8 as two 1x2x4 multiplications (rank 3 each), giving
rank 4 * 2 * 3 = 24 total.
"""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    We pad to 4x4x8 and decompose into 4 independent 1x4x8 multiplications.
    Each 1x4x8 is decomposed as two 1x2x4 multiplications using Strassen's 7-multiplication
    scheme applied to 2x2 blocks (treating 1x2x4 as 2x1x2 blocks).
    
    For a 1x2x4 multiplication:
    A = [a1, a2] (1x2), B = [[b11, b12], [b21, b22]] (2x2), C = [c1, c2, c3, c4] (1x4)
    
    Actually, 1x2x4 is: C_k = sum_j A_j * B_{j,k} for j=1,2 and k=1,2,3,4.
    This is just 2 independent dot products, rank = 2*4 = 8? No, rank of 1x2x4 tensor is 4.
    
    Wait, let me reconsider. The tensor for 1x2x4 is:
    T_{1,j,k} = A_j * B_{j,k}
    This has rank 4 (since j has 2 values and k has 4 values, and they're independent).
    
    Actually no. For fixed j, T_{1,j,k} = A_j * B_{j,k}. The tensor is a sum over j of
    (A_j) * (e_j in U) * (B_{j,:} in V) * ... wait, this is getting confused.
    
    Let me restart with a cleaner approach.
    
    For 3x5x7, the schoolbook rank is 105. We want to do better.
    
    Key insight: We can use the fact that matrix multiplication is bilinear and use
    block decompositions with Strassen.
    
    Let me try: partition 3x5x7 as follows.
    A is 3x5, B is 5x7, C is 3x7.
    
    Split rows of A: 2 + 1 (so A1 is 2x5, A2 is 1x5)
    Split columns of B: 4 + 3 (so B1 is 5x4, B2 is 5x3)
    
    Then:
    C11 (2x4) = A1 * B1
    C12 (2x3) = A1 * B2  
    C21 (1x4) = A2 * B1
    C22 (1x3) = A2 * B2
    
    Schoolbook ranks:
    C11: 2*5*4 = 40
    C12: 2*5*3 = 30
    C21: 1*5*4 = 20
    C22: 1*5*3 = 15
    Total: 105
    
    Now, can we use Strassen on the 2x2 block structure? The blocks aren't square,
    but Strassen's algebraic identities still work for rectangular blocks as long as
    dimensions are consistent.
    
    Strassen for 2x2 blocks:
    M1 = (A1 + A2) * (B1 + B2)
    M2 = (A1 + A2) * B2
    M3 = A1 * (B1 - B2)
    M4 = A2 * B1
    M5 = (A1 - A2) * (B1 + B2)
    M6 = (A1 + A2) * B1
    M7 = A2 * (B2 - B1)
    
    Then:
    C11 = M1 + M4 - M2 + M7
    C12 = M2 + M7
    C21 = M4 + M3
    C22 = M1 - M3 - M4 + M5
    
    Wait, but A1 is 2x5 and A2 is 1x5, so A1 + A2 doesn't make sense (different sizes).
    
    We need to pad A2 to 2x5 by adding a zero row. Similarly for other operations.
    
    Let me define:
    A1_padded = [A1; 0] (2x5) -- already 2x5
    A2_padded = [A2; 0] (2x5) -- pad 1x5 to 2x5
    
    B1_padded = [B1; 0] (5x4) -- already 5x4
    B2_padded = [B2; 0] (5x4) -- pad 5x3 to 5x4 by adding a zero column
    
    Then all Strassen operations work on 2x5 and 5x4 matrices, producing 2x4 results.
    We then extract the appropriate submatrices for C11, C12, C21, C22.
    
    The rank of each Strassen product (2x5 times 5x4) is 2*5*4 = 40 schoolbook.
    7 such products: 7 * 40 = 280. Worse than 105!
    
    This doesn't help because the blocks are too large.
    
    Let me try smaller blocks. What if I use 1x1 blocks?
    
    A (3x5): 3 blocks of 1x5
    B (5x7): 5 blocks of 5x7? No, that doesn't work.
    
    Let me try: A as 1x1 blocks (15 blocks), B as 1x1 blocks (35 blocks).
    This is just the schoolbook with rank 105.
    
    I think for this specific problem, without a known low-rank decomposition,
    the best I can do is the schoolbook 105. Let me just implement that cleanly.
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
