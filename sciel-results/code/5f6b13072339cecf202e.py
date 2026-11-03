"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 tensor decomposition for 3x5x6 matrix multiplication
    using block-wise Strassen and outer-product decompositions.
    
    Split: n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    Blocks and their ranks:
    - (2,2,2): 1 block, rank 7 (Strassen)
    - (2,2,2): 1 block, rank 7 (Strassen)
    - (2,1,2): 2 blocks, rank 4 each = 8
    - (1,2,2): 2 blocks, rank 4 each = 8
    - (1,1,2): 3 blocks, rank 2 each = 6
    Wait, let me recount:
    n splits: [2, 1]
    m splits: [2, 2, 1]
    p splits: [2, 2, 2]
    
    Blocks:
    (2,2,2): rank 7
    (2,2,2): rank 7
    (2,1,2): rank 4
    (1,2,2): rank 4
    (1,2,2): rank 4
    (1,1,2): rank 2
    
    Total: 7+7+4+4+4+2 = 28? That's way too low.
    
    Wait, I need to think about this more carefully.
    
    For block (ni, mj, pk), the rank is R(ni, mj, pk).
    
    R(2,2,2) = 7
    R(2,2,2) = 7
    R(2,1,2) = 4
    R(1,2,2) = 4
    R(1,2,2) = 4
    R(1,1,2) = 2
    
    But wait, there are more blocks. Let me list all combinations:
    ni in {2, 1}, mj in {2, 2, 1}, pk in {2, 2, 2}
    
    (2,2,2), (2,2,2), (2,1,2)
    (2,2,2), (2,2,2), (2,1,2)
    (1,2,2), (1,2,2), (1,1,2)
    (1,2,2), (1,2,2), (1,1,2)
    
    Wait, that's only 8 blocks. Let me be more careful.
    
    ni values: [2, 1] (2 values)
    mj values: [2, 2, 1] (3 values)
    pk values: [2, 2, 2] (3 values)
    
    Total blocks: 2 * 3 * 3 = 18
    
    (2,2,2): R=7
    (2,2,2): R=7
    (2,1,2): R=4
    (2,2,2): R=7
    (2,2,2): R=7
    (2,1,2): R=4
    (1,2,2): R=4
    (1,2,2): R=4
    (1,1,2): R=2
    (1,2,2): R=4
    (1,2,2): R=4
    (1,1,2): R=2
    
    Total: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    Hmm, that's 56. But the previous analysis said 84. Let me reconsider.
    
    Actually, I think I was wrong about some ranks. Let me reconsider:
    R(1,1,2) = 2 (two scalar multiplications)
    R(1,2,2) = 4 (row vector times 2x2 matrix)
    R(2,1,2) = 4 (2x1 column times 1x2 row)
    R(2,2,2) = 7 (Strassen)
    
    So the total should be 56. But that seems too optimistic. Let me verify with a simpler case.
    
    Actually, I think the issue is that when we split, we need to be careful about how the blocks combine.
    
    Let me just implement the schoolbook decomposition for now to ensure correctness, and then try to improve.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition: rank = n*m*p = 90
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
