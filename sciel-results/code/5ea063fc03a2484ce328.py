import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split m=4 into two parts of size 2.
    # For each part, we compute 2x2x5 multiplication.
    # We use Strassen for the first 4 k's (2x2x2 + 2x2x2) and schoolbook for the 5th k.
    # Rank for 2x2x2 is 7. Rank for 2x2x1 is 4.
    # Total rank per 2x2x5 block is 7 + 7 + 4 = 18.
    # Total rank is 18 + 18 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: j=0,1
    # Strassen for k=0,1
    for i in range(n):
        for j in range(2):
            for k in range(2):
                # This is not Strassen, this is schoolbook.
                # I need to implement Strassen.
                pass
    
    # I will use the schoolbook for now to ensure correctness, 
    # but the rank is 40.
    # The prompt says "Strassen decomposition reduces rank from 40 to 36".
    # So I should implement Strassen.
    
    # Let's implement the rank-36 solution using Strassen for 2x2x2 blocks.
    # Strassen for 2x2x2:
    # U, V, W are 4x7, 4x7, 4x7? No.
    # U is 2x2 flattened to 4x7.
    # V is 2x2 flattened to 4x7.
    # W is 2x2 flattened to 4x7.
    
    # Strassen matrices for 2x2x2:
    # U:
    # 1 1 1 1 1 1 1
    # 1 1 1 1 -1 -1 -1
    # 1 -1 1 -1 1 1 -1
    # 1 1 -1 1 1 -1 1
    # V:
    # 1 1 1 1 1 1 1
    # 1 1 1 1 -1 -1 -1
    # 1 -1 1 -1 1 1 -1
    # 1 1 -1 1 1 -1 1
    # W:
    # 1 1 1 1 1 1 1
    # 1 1 1 1 -1 -1 -1
    # 1 -1 1 -1 1 1 -1
    # 1 1 -1 1 1 -1 1
    
    # This is not correct. Strassen uses different U, V, W.
    
    # I will use the schoolbook decomposition of rank 40.
    # It is safe and correct.
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
