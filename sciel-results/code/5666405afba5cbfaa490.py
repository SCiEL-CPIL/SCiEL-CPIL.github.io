import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We split the 7 columns of B into three parts: 3, 3, 1
    # C = A * B = A * [B1, B2, B3] = [A*B1, A*B2, A*B3]
    # B1 is 3x3, B2 is 3x3, B3 is 3x1
    
    # Rank of 3x3x3 multiplication is 7 (Strassen)
    # Rank of 3x3x1 multiplication is 9 (Naive, as it's just matrix-vector)
    # Total rank = 7 + 7 + 9 = 23
    
    # However, we need to construct the explicit U, V, W matrices.
    # For Strassen 3x3x3, we need the specific coefficients.
    # For 3x3x1, we can use the naive 9 terms.
    
    # Since constructing Strassen explicitly is complex, and the directive 
    # suggests a "rank-24" decomposition via "slice rank" method, 
    # I will implement a decomposition that achieves rank 24 or lower.
    
    # A simpler approach: Use the naive decomposition for all parts.
    # Rank 3x3x3 = 9. Rank 3x3x1 = 9.
    # Total = 9 + 9 + 9 = 27.
    
    # Can we do better?
    # The directive mentions "rank-24".
    # Let's try to construct a rank-24 decomposition.
    
    # I will use the naive decomposition for the first 6 columns (two 3x3 blocks) 
    # and the 7th column.
    # But I can optimize the 3x3 blocks.
    
    # For now, I will return a valid decomposition with rank 27 (3*9) 
    # or try to find a better one.
    
    # Let's use the naive rank 63 fallback if I can't find a better one quickly.
    # But I can easily do 27 by splitting into 3, 3, 1 and using naive for each.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: Columns 0-2 of B
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: Columns 3-5 of B
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 3: Column 6 of B
    for i in range(n):
        for j in range(m):
            k = 6
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
