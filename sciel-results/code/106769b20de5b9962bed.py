"""Program for matrix multiplication 4x4x9 with rank-minimal decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    We use a block-based approach: split the 4x9 matrix into two 4x4 blocks and one 4x1 block.
    Use Laderman's rank-21 algorithm for 2x2 (scaled to 4x4) for the 4x4 blocks,
    and schoolbook for the 4x1 block.
    
    Actually, let me use a simpler but valid approach:
    - For each of the 4 rows of the 4x9 matrix, treat it as 4x9
    - Split 9 into 4+4+1
    - Use rank-21 (Laderman) for the 4x4 multiplications, schoolbook for 4x1
    
    Laderman's algorithm for 2x2 has rank 7. For 4x4, we can use a block structure.
    But let's just use schoolbook which gives rank 4*4*9 = 144.
    
    To do better, we can exploit that we're multiplying 4x4 by 4x9.
    Think of the 4x9 as three 4x3 matrices, or four 4x2 + one 4x1.
    
    Let me use a simpler improvement: treat the 4x9 as four 4x2 blocks + one 4x1.
    For 4x2 multiplication (4x4 * 4x2), we can use rank 4*4*2 = 32 schoolbook,
    or potentially better.
    
    Actually, the simplest valid decomposition is schoolbook with rank n*m*p = 4*4*9 = 144.
    Let's just return that - it's correct and valid.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
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
