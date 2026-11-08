DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll use a Strassen-inspired approach for 3x3x5 matrix multiplication
    # The goal is to find a low-rank decomposition
    # 
    # For 3x3x5, we can use a hybrid approach:
    # - Split the 3x5 matrix into a 3x3 and a 3x2 part
    # - Use Strassen for the 3x3 part (rank 7)
    # - Use schoolbook for the 3x2 part (rank 6)
    # - Total: 7 + 6 = 13
    #
    # But let's try to do better by finding a more clever decomposition
    # 
    # Actually, let's start with a safe approach: use the schoolbook method
    # but try to find a lower rank decomposition through algebraic manipulation
    #
    # For now, let's implement a basic Strassen-based approach
    # 
    # Strassen for 3x3 matrices requires rank 7 (using 2x2 blocks)
    # But for 3x3, we need to be careful about the block structure
    #
    # Let's use a different approach: direct algebraic construction
    # 
    # The matrix multiplication tensor for (3,3,5) has dimension 3*3*5 = 45
    # We want to find the minimal rank decomposition
    #
    # Let's try to construct a rank-15 decomposition using linear combinations
    # 
    # Actually, let's start with a known result: for 3x3x5, the rank is at most 15
    # (this can be achieved using a combination of Strassen and schoolbook)
    #
    # Let's implement a rank-15 decomposition
    
    # We'll use the following approach:
    # - For the 3x3 part (first 3 columns of the 3x5 matrix), use Strassen (rank 7)
    # - For the 3x2 part (last 2 columns), use schoolbook (rank 6)
    # - Total rank: 7 + 6 = 13
    
    # But we need to be careful about how we combine these
    # Let's just use schoolbook for now as a safe fallback, but with a lower rank
    # 
    # Actually, let's try to find a rank-12 decomposition
    # 
    # For simplicity, let's use the schoolbook method but optimize the construction
    # The schoolbook rank is 45, but we can do better
    #
    # Let's implement a basic optimization: group by the middle index j
    # and use linear combinations to reduce the rank
    
    # For a proper improvement, we need to find a low-rank decomposition
    # Let's use a known result: rank(3,3,5) <= 15
    
    # Construct a rank-15 decomposition
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll use a Strassen-inspired construction
    # For the 3x3 part (columns 0-2), use Strassen-like combinations
    # For the 3x2 part (columns 3-4), use schoolbook
    
    # Strassen for 3x3: we can use 7 rank-1 terms
    # Let's define the Strassen matrices for 3x3
    
    # Actually, for 3x3, we need a different approach
    # Let's use the following:
    # - Split 3x3 into 2x2 blocks with overlap
    # - Use Strassen for the 2x2 parts
    
    # This is getting complex. Let's use a simpler approach:
    # Use schoolbook but with a clever indexing to reduce rank
    
    # For now, let's return a rank-45 schoolbook decomposition as a safe baseline
    # and note that we need to find a better decomposition
    
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
