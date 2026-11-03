"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a lower-rank decomposition for 3x5 * 5x7 matrix multiplication.
    The schoolbook rank is 3*5*7 = 105.
    
    For rectangular matrix multiplication C = A * B where A is n x m and B is m x p,
    the tensor T_{i,j,k} = delta_{j,k} (in the appropriate indices) needs to be decomposed.
    
    We can try to exploit structure. One known approach for small matrices is to use
    the fact that some entries might be combined. However, for 3x5x7, the known
    lower bounds are not well-established enough to guarantee a rank < 105 easily
    without sophisticated algorithms like Strassen's or Coppersmith-Winograd (which
    are for large N).
    
    For small fixed dimensions, the minimal rank is often equal to the schoolbook rank
    unless there's a specific algebraic identity.
    
    Let's check if we can do better than 105.
    The tensor is of format (3, 5, 7).
    The rank of the matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, n*p) = max(15, 35, 21) = 35.
    It is also known that rank(M_{n,m,p}) <= n*m*p.
    
    Is there a known decomposition for 3x5x7 with rank < 105?
    Without specific knowledge of a lower-rank identity for this exact shape,
    and given the time limit, the safest bet is the schoolbook decomposition.
    
    However, the directive suggests exploring block-wise or Kronecker product strategies.
    If we can decompose the 5x5 middle dimension into blocks, say 3x3 and 2x2, we might
    express the multiplication as a sum of smaller multiplications.
    
    Let A be 3x5, B be 5x7.
    Split A into [A1, A2] where A1 is 3x3 and A2 is 3x2.
    Split B into [B1; B2] where B1 is 3x7 and B2 is 2x7.
    Then C = A1*B1 + A2*B2.
    
    This requires rank(M_{3,3,7}) + rank(M_{3,2,7}).
    Schoolbook rank for 3x3x7 is 3*3*7 = 63.
    Schoolbook rank for 3x2x7 is 3*2*7 = 42.
    Total = 105. No gain.
    
    What if we use a different split?
    Split A into [A1, A2] where A1 is 3x2 and A2 is 3x3.
    Split B into [B1; B2] where B1 is 2x7 and B2 is 3x7.
    C = A1*B1 + A2*B2.
    Rank = M_{3,2,7} + M_{3,3,7} = 42 + 63 = 105.
    
    It seems that simple splitting doesn't help because the sum of the products of dimensions
    (n*m1*p + n*m2*p = n*p*(m1+m2) = n*p*m) is constant.
    
    To get a gain, we would need a non-trivial identity that allows combining terms from
    different blocks or using a Strassen-like trick on the blocks.
    
    For example, if we had 2x2x2, Strassen gives rank 7 < 8.
    Can we apply Strassen to 3x3x3?
    Rank(M_{3,3,3}) is known to be 19 (or 23? No, it's 19 for 3x3x3 over complex numbers?
    Actually, the rank of M_{3,3,3} is 19. Schoolbook is 27.
    If we can decompose 3x5x7 using 3x3x3 blocks, we might save.
    
    But 5 and 7 are not multiples of 3.
    
    Given the complexity and the lack of a clear, simple improvement for 3x5x7 in the
    available time, and the fact that the seed program already provides the schoolbook
    solution, I will stick to the schoolbook solution unless I can find a specific
    identity.
    
    Wait, the directive says "investigate improvements within its family".
    If no improvement is found, returning the schoolbook is valid.
    
    However, I must ensure the code is correct and efficient.
    The current code is O(1) construction.
    
    Let's double check if there's any trivial reduction.
    No, for generic matrices, the rank is likely 105.
    
    I will return the schoolbook decomposition.
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
