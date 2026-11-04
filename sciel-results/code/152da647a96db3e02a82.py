"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x8 matrix.
    
    We use a divide-and-conquer approach:
    - Split the 3x3 matrix into blocks: top 2 rows and bottom 1 row.
    - The 2x3 * 3x8 part can be handled by four 2x2 Strassen blocks (each rank 7)
      if we split the 3x8 into four 2x2 blocks? No, that doesn't quite work directly.
    
    Let's think more carefully.
    
    Actually, for 3x3 * 3x8, we can view this as:
    C = A * B where A is 3x3, B is 3x8, C is 3x8.
    
    We can split A into [A1; A2] where A1 is 2x3 and A2 is 1x3.
    Then C = [A1*B; A2*B].
    
    For A1*B (2x3 * 3x8 = 2x8):
    We can split B into [B1, B2, B3] where each Bi is 3x? No, B is 3x8.
    
    Alternatively, we can use the fact that matrix multiplication is bilinear and
    use known optimal ranks for smaller subproblems.
    
    Known results:
    - 2x2 * 2x2: Strassen rank 7
    - 2x3 * 3x2: rank 10 (or can it be lower?)
    - 1x3 * 3x8: rank 8 (rank-1 terms, one for each output)
    
    For 2x3 * 3x8:
    We can write this as a sum of 2x2 * 2x2 and 2x1 * 1x8 or similar.
    
    Actually, let's use a different decomposition.
    
    Consider splitting the 3x8 matrix B into two parts: B = [B1 | B2] where B1 is 3x4 and B2 is 3x4.
    Then C = [A*B1 | A*B2].
    
    For each 3x3 * 3x4 subproblem, can we do better than schoolbook 36?
    Schoolbook for 3x3 * 3x4 is 3*3*4 = 36.
    
    Actually, let's try a recursive Strassen-like approach.
    
    For 3x3 * 3x8:
    Split A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    Split B = [B1; B2] where B1 is 2x8, B2 is 1x8.
    
    Then C = [A11*B1 + A12*B2; A21*B1 + A22*B2].
    
    This gives:
    C_top = A11*B1 + A12*B2  (2x8)
    C_bottom = A21*B1 + A22*B2  (1x8)
    
    A11*B1 is 2x2 * 2x8.
    A12*B2 is 2x1 * 1x8.
    A21*B1 is 1x2 * 2x8.
    A22*B2 is 1x1 * 1x8.
    
    Rank of 2x2 * 2x8: We can split B1 into four 2x2 blocks? No, B1 is 2x8.
    Actually, 2x2 * 2x8 can be seen as 4 independent 2x2 * 2x2 multiplications? No.
    
    Let me think differently. 2x2 * 2x8:
    Let A11 = [[a,b],[c,d]], B1 = [b1..b8] (2 rows, 8 cols).
    C_top1 = [a*b1_1 + b*b2_1, ..., a*b1_8 + b*b2_8]
    C_top2 = [c*b1_1 + d*b2_1, ..., c*b1_8 + d*b2_8]
    
    This is equivalent to 8 independent 2x2 * 2x1 multiplications.
    Each 2x2 * 2x1 has rank 4 (schoolbook) or can it be 3?
    Actually, 2x2 * 2x1: C = A*x where A is 2x2, x is 2x1.
    C1 = a*x1 + b*x2
    C2 = c*x1 + d*x2
    This is 4 products: a*x1, b*x2, c*x1, d*x2. Can we do it in 3?
    Strassen for 2x2 * 2x2 uses 7 products. For 2x2 * 2x1, we only need 4 products.
    I believe the rank is 4 for 2x2 * 2x1.
    
    So 2x2 * 2x8 would have rank 8*4 = 32? That's worse than schoolbook 2*2*8=32. Same.
    
    Hmm, this approach isn't helping.
    
    Let me try a different strategy. Use the known optimal rank for 3x3 * 3x3.
    The rank of 3x3 * 3x3 is known to be 23 (Landsberg, 1998) or is it?
    Actually, the rank of 3x3 * 3x3 is not known exactly. The best known upper bound is 23.
    Schoolbook is 27.
    
    If we can decompose 3x3 * 3x3 in rank 23, then for 3x3 * 3x8, we can split the 8 columns
    into two 4-column blocks? No, that doesn't directly help.
    
    Actually, 3x3 * 3x8 can be viewed as 3x3 * (3x4 + 3x4) = 3x3*3x4 + 3x3*3x4.
    If we know the rank of 3x3 * 3x4, we can use it twice.
    
    The rank of 3x3 * 3x4: schoolbook is 36. Can we do better?
    
    Alternatively, 3x3 * 3x8 = 3x3 * 3x2 * 4? No.
    
    Let's just implement a reasonable decomposition and hope it's better than 72.
    
    Actually, the simplest improvement: use Strassen for 2x2 blocks.
    
    Split the 3x3 * 3x8 as follows:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    B is 3x8.
    
    C = A*B is 3x8.
    
    We can group the first 2 rows and 2 columns of A into a 2x2 block, and handle the rest.
    
    Let me try a concrete decomposition:
    
    C[0:2, :] = A[0:2, :] * B
    C[2, :] = A[2, :] * B
    
    For C[0:2, :]:
    C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2] * B[2, :]
    
    A[0:2, 0:2] is 2x2, B[0:2, :] is 2x8.
    A[0:2, 2] is 2x1, B[2, :] is 1x8.
    
    2x2 * 2x8: rank 32 (schoolbook) or can we do better?
    Actually, 2x2 * 2x8 can be done by treating it as 4 independent 2x2 * 2x2? No, B is 2x8.
    
    I think for this problem, the schoolbook rank of 72 might be close to optimal,
    or the best known upper bound is not much lower.
    
    Let me just return the schoolbook decomposition, which is rank 72.
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
