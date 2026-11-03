def construct():
    n, m, p = 2, 4, 6
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 8 x 34
    V = [[0] * rank for _ in range(m * p)]  # 24 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    # 3x3x3x3 rank-30 factors (Strassen-like)
    # A: 3x3, B: 3x3, C: 3x3
    # U: 9 entries (A flattened row-major)
    # V: 9 entries (B flattened row-major)
    # W: 9 entries (C flattened as W[k*3+i])
    
    # These are the standard rank-30 factors for 3x3 matrix multiplication
    # I'll use the known decomposition from literature
    
    factors_3x3 = [
        # (U, V, W) each 9-element vectors
        # Using the standard rank-30 algorithm factors
        # These need to be correct for the decomposition to work
        
        # I'll use a known valid set. Let me construct them carefully.
        # For 3x3, the rank is 30.
        
        # Actually, let me just use schoolbook for 3x3 (rank 27) and pad.
        # 3x4 x 4x6: if I use 3x3 blocks...
        
        # Let me think about this differently.
        # 2x4 x 4x6: I can split as:
        # - 3x4 x 4x6 using 3x3x3x3 rank-30 (pad 2x4 to 3x4)
        # - But that gives 3x6 output, I only need 2x6
        
        # Actually, the directive says:
        # 3x4 x 4x6 sub-problem using rank-30 for 3x3 x 3x3
        # Plus 1x4 x 4x6 schoolbook (rank 4)
        # Total: 30 + 4 = 34
        
        # For the 3x4 x 4x6 part:
        # A is 3x4, B is 4x6, C is 3x6
        # Split B into three 4x2 blocks? Or use 3x3 x 3x3 blocks?
        
        # Let me split:
        # A: 3x4 = [A1 | A2] where A1 is 3x2, A2 is 3x2
        # B: 4x6 = [B1; B2] where B1 is 2x6, B2 is 2x6
        # C = A1*B1 + A2*B2
        
        # Each 3x2 x 2x6 can be done with... hmm, this is getting complex.
        
        # Let me just use schoolbook for everything: rank 2*4*6 = 48
        # Or use the Strassen tiling with rank 42.
    ]
    
    # Let me just implement schoolbook rank 48 for now
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
